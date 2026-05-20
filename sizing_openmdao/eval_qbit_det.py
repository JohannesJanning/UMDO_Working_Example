"""
run_qbit_deterministic.py - Deterministic QBiT evaluation driver.

Evaluates the QBiT sizing model at a fixed design point (no optimization).
Given design variables [V_inf, r, J, S_w] and fixed t_hover,
returns the resulting MTOM and constraint values.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import openmdao.api as om
from scipy.optimize import brentq

om.config_reports = False

from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (AR_FIXED, BATTERY_EFF, G,
                             W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
                             J_BOUNDS, S_W_BOUNDS, DL_MAX, BL_MAX, CL_MAX)
import qbit.components.sizing_comps as sc


# ============================================================================
# USER INPUTS - MODIFY THESE
# ============================================================================

# Mission parameters
PAYLOAD_KG = 3.0          # Payload mass [kg]
RANGE_M = 15000.0         # Mission range [m] (one-way)
N_CUSTOMERS = 2           # Number of customers

# Design point to evaluate
V_INF = 29.72              # Cruise speed [m/s]
R = 0.2652                # Rotor radius [m]
J = 1.300                 # Propeller advance ratio [-]
S_W = 0.2415              # Wing area [m²]

# Hover time setting (choose one)
T_HOVER = 101.0            # Mean hover time [s] (typical mission)
# T_HOVER = 101.0         # 97.5th percentile [s] (conservative/worst-case)

# Verbose output
VERBOSE = True

# ============================================================================


@dataclass
class DeterministicEvaluationResult:
    """Results from deterministic evaluation of a fixed design point."""
    V_inf: float
    r: float
    J: float
    S_w: float
    t_hover: float
    W_total: float
    W_battery: float
    W_empty: float
    P_hover: float
    P_cruise: float
    b: float
    chord: float
    E_req: float
    disk_loading: float
    blade_loading: float
    cruise_CL: float
    weight_residual: float
    converged: bool
    solve_time: float
    DL_MAX: float
    BL_MAX: float
    CL_MAX: float
    
    @property
    def MTOM_kg(self) -> float:
        return self.W_total / G
    
    def summary(self) -> str:
        dl_margin = self.DL_MAX - self.disk_loading
        bl_margin = self.BL_MAX - self.blade_loading
        cl_margin = self.CL_MAX - self.cruise_CL
        
        lines = [
            "=" * 60,
            "DETERMINISTIC EVALUATION RESULTS",
            "=" * 60,
            "Input Design Point:",
            f"  • Cruise speed V_inf : {self.V_inf:7.2f} m/s",
            f"  • Rotor radius r     : {self.r:7.4f} m",
            f"  • Advance ratio J    : {self.J:7.3f}",
            f"  • Wing area S_w      : {self.S_w:7.4f} m²",
            f"  • Hover time t_hover : {self.t_hover:7.1f} s",
            "",
            "Output Performance:",
            f"  • MTOM               : {self.MTOM_kg:7.3f} kg  ({self.W_total:.1f} N)",
            f"  • Battery mass       : {self.W_battery/G:7.3f} kg",
            f"  • Empty mass         : {self.W_empty/G:7.3f} kg",
            f"  • Wingspan           : {self.b:7.4f} m  (AR={AR_FIXED})",
            f"  • Mean chord         : {self.chord:7.4f} m",
            f"  • P_hover            : {self.P_hover:8.1f} W",
            f"  • P_cruise           : {self.P_cruise:8.1f} W",
            f"  • E_required         : {self.E_req/3600:.3f} Wh",
            "",
            "Constraints & Margins:",
            f"  • Weight Residue     : {self.weight_residual:10.4e} (Goal: 0.0)",
            f"  • Disk loading       : {self.disk_loading:7.2f} / {self.DL_MAX} N/m² (Margin: {dl_margin:7.2f})",
            f"  • Blade loading      : {self.blade_loading:7.4f} / {self.BL_MAX}      (Margin: {bl_margin:7.4f})",
            f"  • Cruise CL          : {self.cruise_CL:7.4f} / {self.CL_MAX}      (Margin: {cl_margin:7.4f})",
            "",
            f"Status: {'✓ Converged' if self.converged else '✗ Failed'}",
            f"Solve time: {self.solve_time:.2f} s",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def check_constraints(self) -> bool:
        """Check if all constraints are satisfied."""
        dl_ok = self.disk_loading <= self.DL_MAX
        bl_ok = self.blade_loading <= self.BL_MAX
        cl_ok = self.cruise_CL <= self.CL_MAX
        weight_ok = abs(self.weight_residual) < 1e-3
        
        all_ok = dl_ok and bl_ok and cl_ok and weight_ok
        
        print("\nConstraint Check:")
        print(f"  Disk loading:  {self.disk_loading:.2f} ≤ {self.DL_MAX} → {'✓' if dl_ok else '✗'}")
        print(f"  Blade loading: {self.blade_loading:.4f} ≤ {self.BL_MAX} → {'✓' if bl_ok else '✗'}")
        print(f"  Cruise CL:     {self.cruise_CL:.4f} ≤ {self.CL_MAX} → {'✓' if cl_ok else '✗'}")
        print(f"  Weight closure: |{self.weight_residual:.2e}| < 1e-3 → {'✓' if weight_ok else '✗'}")
        
        return all_ok


class DeterministicEvaluator:
    """Deterministic evaluator for QBiT model at fixed design point."""
    
    def __init__(self, payload_kg: float, range_m: float, n_c: int):
        self.payload_kg = payload_kg
        self.range_m = range_m
        self.n_c = n_c
        
    def _build_problem(self) -> om.Problem:
        prob = om.Problem(reports=None)
        prob.model = build_qbit_model(self.payload_kg, self.range_m, self.n_c)
        
        prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
        prob.model.set_input_defaults("V_inf", val=33.0, units="m/s")
        prob.model.set_input_defaults("r", val=0.22, units="m")
        prob.model.set_input_defaults("J", val=1.3)
        prob.model.set_input_defaults("S_w", val=0.20, units="m**2")
        
        prob.setup()
        return prob
    
    def _solve_for_Wtotal(self, prob: om.Problem, t_hover: float, 
                          V_inf: float, r: float, J: float, S_w: float):
        """Solve for W_total using root-finding."""
        _orig_t = getattr(sc, "T_HOVER", None)
        sc.T_HOVER = float(t_hover)
        
        try:
            def eval_res(W: float) -> float:
                try:
                    prob.set_val("W_total", W)
                    prob.set_val("V_inf", V_inf)
                    prob.set_val("r", r)
                    prob.set_val("J", J)
                    prob.set_val("S_w", S_w)
                    prob.run_model()
                    return float(prob.get_val("weight_residual")[0])
                except Exception:
                    return float("nan")
            
            wl, wh = float(W_TOTAL_BOUNDS[0]), float(W_TOTAL_BOUNDS[1])
            rl, rh = eval_res(wl), eval_res(wh)
            
            if np.isnan(rl) or np.isnan(rh) or rl * rh > 0:
                found = False
                for xl, xr in zip(np.linspace(wl, wh, 25)[:-1], np.linspace(wl, wh, 25)[1:]):
                    fl, fr = eval_res(xl), eval_res(xr)
                    if not (np.isnan(fl) or np.isnan(fr)) and fl * fr <= 0:
                        wl, wh = xl, xr
                        found = True
                        break
                if not found:
                    return None
            
            return brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)
        except Exception:
            return None
        finally:
            if _orig_t is None:
                if hasattr(sc, "T_HOVER"):
                    del sc.T_HOVER
            else:
                sc.T_HOVER = _orig_t
    
    def evaluate(self, V_inf: float, r: float, J: float, S_w: float, 
                 t_hover: float, verbose: bool = True) -> DeterministicEvaluationResult:
        """Evaluate QBiT at a fixed design point."""
        import time
        start_time = time.time()
        
        prob = self._build_problem()
        W_total = self._solve_for_Wtotal(prob, t_hover, V_inf, r, J, S_w)
        
        if W_total is None:
            if verbose:
                print("Root-finding failed, attempting driver-based solve...")
            
            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['tol'] = 1e-6
            prob.driver.options['maxiter'] = 500
            
            prob.model.add_design_var('W_total', lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
            prob.model.add_objective('W_total')
            prob.model.add_constraint('weight_residual', equals=0.0)
            
            prob.setup()
            prob.set_val('W_total', 6.0 * G)
            prob.set_val('V_inf', V_inf)
            prob.set_val('r', r)
            prob.set_val('J', J)
            prob.set_val('S_w', S_w)
            
            _orig_t = getattr(sc, "T_HOVER", None)
            sc.T_HOVER = float(t_hover)
            try:
                prob.run_driver()
                W_total = float(prob.get_val('W_total')[0])
                converged = getattr(getattr(prob.driver, 'result', None), 'success', False)
            except Exception:
                W_total = W_TOTAL_BOUNDS[1]
                converged = False
            finally:
                if _orig_t is None:
                    if hasattr(sc, "T_HOVER"):
                        del sc.T_HOVER
                else:
                    sc.T_HOVER = _orig_t
        else:
            _orig_t = getattr(sc, "T_HOVER", None)
            sc.T_HOVER = float(t_hover)
            try:
                prob.set_val("W_total", W_total)
                prob.set_val("V_inf", V_inf)
                prob.set_val("r", r)
                prob.set_val("J", J)
                prob.set_val("S_w", S_w)
                prob.run_model()
                converged = True
            except Exception:
                converged = False
            finally:
                if _orig_t is None:
                    if hasattr(sc, "T_HOVER"):
                        del sc.T_HOVER
                else:
                    sc.T_HOVER = _orig_t
        
        solve_time = time.time() - start_time
        
        if converged:
            W_battery = float(prob.get_val('W_battery')[0])
            W_empty = float(prob.get_val('W_empty')[0])
            P_hover = float(prob.get_val('P_hover')[0]) / BATTERY_EFF
            P_cruise = float(prob.get_val('P_cruise')[0]) / BATTERY_EFF
            E_req = float(prob.get_val('E_req')[0])
            disk_loading = float(prob.get_val('disk_loading')[0])
            blade_loading = float(prob.get_val('blade_loading')[0])
            cruise_CL = float(prob.get_val('cruise_CL')[0])
            weight_residual = float(prob.get_val('weight_residual')[0])
            b = float(np.sqrt(AR_FIXED * S_w))
            chord = S_w / b
        else:
            W_battery = W_empty = P_hover = P_cruise = E_req = 0.0
            disk_loading = blade_loading = cruise_CL = weight_residual = float('nan')
            b = float(np.sqrt(AR_FIXED * S_w))
            chord = S_w / b if b > 0 else 0.0
        
        result = DeterministicEvaluationResult(
            V_inf=V_inf, r=r, J=J, S_w=S_w, t_hover=t_hover,
            W_total=W_total, W_battery=W_battery, W_empty=W_empty,
            P_hover=P_hover, P_cruise=P_cruise, b=b, chord=chord, E_req=E_req,
            disk_loading=disk_loading, blade_loading=blade_loading,
            cruise_CL=cruise_CL, weight_residual=weight_residual,
            converged=converged, solve_time=solve_time,
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )
        
        if verbose:
            print(result.summary())
        
        return result


if __name__ == "__main__":
    # Create evaluator with mission settings
    evaluator = DeterministicEvaluator(
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS
    )
    
    # Run evaluation
    result = evaluator.evaluate(
        V_inf=V_INF,
        r=R,
        J=J,
        S_w=S_W,
        t_hover=T_HOVER,
        verbose=VERBOSE
    )
    
    # Check constraints
    result.check_constraints()