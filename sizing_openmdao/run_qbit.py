"""
run_qbit.py - QBiT sizing optimization driver.
- core deterministic optimization script

"""

#####################################
# 1. Imports & Configurations
#####################################
# - Sets up the Python environment
# - Imports necessary libraries (numpy, openmdao, dataclasses)
# - Imports the QBiT aircraft model and physical constants


from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import openmdao.api as om
om.config_reports = False
from scipy.optimize import brentq
import qbit.components.sizing_comps as sc

from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (AR_FIXED, BATTERY_EFF, G,
                             W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
                             J_BOUNDS, S_W_BOUNDS, DL_MAX, BL_MAX, CL_MAX)







#####################################
# 2. Data Container - SizingResult
#####################################
# - Defines a structured container for optimization results
# - Provides a summary() method to print formatted output


@dataclass
class SizingResult:
    W_total: float; W_battery: float; W_empty: float
    P_hover: float; P_cruise: float
    V_inf: float; r: float; J: float; S_w: float
    b: float; chord: float; E_req: float; converged: bool
    disk_loading: float; blade_loading: float; cruise_CL: float; weight_residual: float
    DL_MAX: float; BL_MAX: float; CL_MAX: float

    def summary(self) -> str:
        # Calculate margins (Difference between limit and actual)
        dl_margin = self.DL_MAX - self.disk_loading
        bl_margin = self.BL_MAX - self.blade_loading
        cl_margin = self.CL_MAX - self.cruise_CL
        lines = [
            f"  MTOM          : {self.W_total/G:7.3f} kg  ({self.W_total:.1f} N)",
            f"  Battery mass  : {self.W_battery/G:7.3f} kg",
            f"  Empty mass    : {self.W_empty/G:7.3f} kg",
            f"  Cruise speed  : {self.V_inf:7.2f} m/s",
            f"  Rotor radius  : {self.r:7.4f} m",
            f"  Wing area     : {self.S_w:7.4f} m²",
            f"  Wingspan      : {self.b:7.4f} m  (AR={AR_FIXED})",
            f"  Mean chord    : {self.chord:7.4f} m",
            f"  Prop adv. J   : {self.J:5.3f}",
            f"  P_hover       : {self.P_hover:8.1f} W",
            f"  P_cruise      : {self.P_cruise:8.1f} W",
            f"  E_required    : {self.E_req/3600:.3f} Wh",
            f"  Converged     : {self.converged}",
            "--- Constraints & Margins ---",
            f"  Weight Resid. : {self.weight_residual:10.4e} (Goal: 0.0)",
            f"  Disk Loading  : {self.disk_loading:7.2f} / {self.DL_MAX} N/m² (Margin: {dl_margin:7.2f})",
            f"  Blade Loading : {self.blade_loading:7.4f} / {self.BL_MAX}      (Margin: {bl_margin:7.4f})",
            f"  Cruise CL     : {self.cruise_CL:7.4f} / {self.CL_MAX}      (Margin: {cl_margin:7.4f})",
        ]
        return "\n".join(lines)









#####################################
# 3. Problem Builder - build_problem()
#####################################
# - Assembling the optimization problem
# - Connects physics model -> optimizer -> constraints
# - Defines design variables, objective, and constraints



def build_problem(payload_kg: float, range_m: float, n_c: int = 1) -> om.Problem:
    prob = om.Problem(reports=None)
    prob.model = build_qbit_model(payload_kg, range_m, n_c) # imported from qbit.models.qbit_model


    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP' # Sequential Least Squares Programming
    prob.driver.options['tol']       = 1e-9 # Convergence tolerance
    prob.driver.options['maxiter']   = 2000 # Maximum iterations

    # Design variables: W_total is also a design var (SLSQP controls it
    # together with V_inf, r, J, S_w to satisfy weight_residual=0)
    # bounds defined in qbit.constants
    prob.model.add_design_var('W_total', lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1]) # total weight
    prob.model.add_design_var('V_inf',   lower=V_INF_BOUNDS[0],   upper=V_INF_BOUNDS[1]) # cruise speed
    prob.model.add_design_var('r',       lower=R_BOUNDS[0],       upper=R_BOUNDS[1]) # rotor radius
    prob.model.add_design_var('J',       lower=J_BOUNDS[0],       upper=J_BOUNDS[1]) # propeller advance ratio
    prob.model.add_design_var('S_w',     lower=S_W_BOUNDS[0],     upper=S_W_BOUNDS[1]) # wing area

    prob.model.add_objective('W_total') # Minimize MTOM

    # Weight closure equality + aerodynamic/structural inequalities
    prob.model.add_constraint('weight_residual', equals=0.0) #defined in qbit.components.weight_balance.py
    prob.model.add_constraint('disk_loading',    upper=DL_MAX) # defined in qbit.groups.constraints_group.py
    prob.model.add_constraint('blade_loading',   upper=BL_MAX)
    prob.model.add_constraint('cruise_CL',       upper=CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N") # intial value 
    prob.setup() # finalize problem setup 
    prob.set_val('W_total', 6.0 * G) # intial runtime value for W_total
    return prob






#####################################
# 4. Fixed-design Evaluator for UQ
#####################################
# - Evaluates QBiT at fixed V_inf, r, J, S_w
# - Solves W_total from weight_residual = 0
# - Used by UQPCE and Monte Carlo wrappers

def evaluate_qbit_at_design(
    prob: om.Problem,
    t_hover: float,
    dvars: tuple[float, float, float, float],
    verbose: bool = False
) -> dict | None:
    """
    Evaluate QBiT at fixed design variables.

    V_inf, r, J, and S_w are fixed.
    W_total is solved from weight_residual = 0.
    Used by UQ, PCE, and Monte Carlo wrappers.
    """
    V, r, J, Sw = dvars

    _orig_t = getattr(sc, "T_HOVER", None)
    sc.T_HOVER = float(t_hover)

    try:
        def eval_res(W: float) -> float:
            try:
                prob.set_val("W_total", W)
                prob.set_val("V_inf", V)
                prob.set_val("r", r)
                prob.set_val("J", J)
                prob.set_val("S_w", Sw)
                prob.run_model()
                return float(prob.get_val("weight_residual")[0])
            except Exception:
                return float("nan")

        wl = float(W_TOTAL_BOUNDS[0])
        wh = float(W_TOTAL_BOUNDS[1])

        rl, rh = eval_res(wl), eval_res(wh)

        if np.isnan(rl) or np.isnan(rh) or rl * rh > 0:
            found = False
            grid = np.linspace(wl, wh, 25)

            for xl, xr in zip(grid[:-1], grid[1:]):
                fl, fr = eval_res(xl), eval_res(xr)

                if not (np.isnan(fl) or np.isnan(fr)) and fl * fr <= 0:
                    wl, wh = xl, xr
                    found = True
                    break

            if not found:
                raise om.AnalysisError(
                    f"No sign change in weight residual for t_hover={t_hover:.1f}s"
                )

        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)

        # Ensure outputs correspond to converged root
        prob.set_val("W_total", root)
        prob.set_val("V_inf", V)
        prob.set_val("r", r)
        prob.set_val("J", J)
        prob.set_val("S_w", Sw)
        prob.run_model()

        return {
            "W_total": float(root),
            "cruise_CL": float(prob.get_val("cruise_CL")[0]),
            "disk_loading": float(prob.get_val("disk_loading")[0]),
            "blade_loading": float(prob.get_val("blade_loading")[0]),
            "weight_residual": float(prob.get_val("weight_residual")[0]),
        }

    except Exception as exc:
        if verbose:
            print(f"  [evaluate_qbit_at_design] FAILED t_hover={t_hover:.1f}s: {exc}")
        return None

    finally:
        if _orig_t is None:
            if hasattr(sc, "T_HOVER"):
                del sc.T_HOVER
        else:
            sc.T_HOVER = _orig_t







#####################################
# 5. Solver Wrapper - Stage1Problem
#####################################
# - Wrapping the optimization in a reusable class
# - Handles result extraction and formatting


class Stage1Problem:
    #Step 1: Object Initialization (__init__), creating a new object
    def __init__(self, payload_kg: float, range_m: float, n_c: int = 1):
        self.payload_kg = payload_kg
        self.range_m    = range_m
        self.n_c        = n_c


    # Add to Stage1Problem class in run_qbit.py:
    def _build_problem(self) -> om.Problem:
        """Return the OpenMDAO problem without running it."""
        return build_problem(self.payload_kg, self.range_m, self.n_c)

    #Step 2: Solving the optimization problem (solve), running the optimization and extracting results
    def solve(self, verbose: bool = True) -> SizingResult:
        prob = build_problem(self.payload_kg, self.range_m, self.n_c)
        # Step 3: Print Header (if verbose)
        if verbose:
            print("=" * 60)
            print("QBiT OpenMDAO Model")
            print(f"Single UAV - {self.n_c} Node(s) - {self.range_m*2/1000:.1f} km Range - Minimise MTOM")
            print("=" * 60)
        # Step 4: Run the Optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_driver()
        # Step 5: Extract Results from the Problem
        W  = float(prob.get_val('W_total')[0])
        Ph = float(prob.get_val('P_hover')[0] / BATTERY_EFF)
        Pc = float(prob.get_val('P_cruise')[0] / BATTERY_EFF)
        V  = float(prob.get_val('V_inf')[0])
        r  = float(prob.get_val('r')[0])
        J  = float(prob.get_val('J')[0])
        Sw = float(prob.get_val('S_w')[0])
        Wb = float(prob.get_val('W_battery')[0])
        We = float(prob.get_val('W_empty')[0])
        E  = float(prob.get_val('E_req')[0])
        # Step 6: Calculate Derived Quantities
        b  = float(np.sqrt(AR_FIXED * Sw))
        dl = float(prob.get_val('disk_loading')[0])
        bl = float(prob.get_val('blade_loading')[0])
        cl = float(prob.get_val('cruise_CL')[0])
        wr = float(prob.get_val('weight_residual')[0])
        # Step 7: Check Convergence Status
        converged = getattr(getattr(prob.driver, 'result', None), 'success', True)
        # Step 8: Package Results into SizingResult
        result = SizingResult(
            W_total=W, W_battery=Wb, W_empty=We,
            P_hover=Ph, P_cruise=Pc, V_inf=V, r=r, J=J, S_w=Sw,
            b=b, chord=Sw/b, E_req=E, converged=converged, disk_loading=dl, blade_loading=bl, cruise_CL=cl, weight_residual=wr,
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )
        # Step 9: Print Design Vector
        print("\nFull precision design vector [V_inf, r, J, S_w, W_total]:")
        print(repr([
            float(prob.get_val('V_inf')[0]),
            float(prob.get_val('r')[0]),
            float(prob.get_val('J')[0]),
            float(prob.get_val('S_w')[0]),
        ]))
        # Step 10: Print Detailed Summary
        if verbose:
            print(result.summary())
            print("=" * 60)
        return result




#####################################
# 6. Main execution 
#####################################
# - script entry point 

if __name__ == '__main__':
    # 1. Create problem and solve it
    res = Stage1Problem(payload_kg=3.0, range_m=15_000.0, n_c=2).solve(verbose=True)
    # 2. Convert weight from Newtons to kg (divide by gravity)
    mtom_kg = res.W_total / G
    # 3. Sanity check: MTOM should be between 0.5 and 50 kg
    assert 0.5 <= mtom_kg <= 50.0, f"MTOM {mtom_kg:.3f} kg outside [0.5, 50.0]"
    # 4. Check that optimizer converged
    assert res.converged
    # 5. Print success message
    print(f"\nValidation passed: MTOM = {mtom_kg:.3f} kg ✓")



#####################################
# Remarks on Results:
# 
# Iterations: 
# - Number of major optimization iterations where SLSQP updated all design variables together.
# -- Each iteration:
# ----Evaluates objective and constraints at current point
# ----Computes gradients (derivatives)
# ----Solves a quadratic subproblem
# ----Takes a step to new design point
#
# Function evaluations:
#  - Number of times the model was evaluated (objective + constraints)
# 
# Gradient evaluations:
# - Number of times derivatives were computed
