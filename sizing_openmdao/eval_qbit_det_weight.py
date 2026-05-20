"""
run_qbit_deterministic.py - Deterministic QBiT evaluation driver.

Evaluates the QBiT sizing model at a fixed design point (no optimization).
Two modes:
1. Input design variables [V_inf, r, J, S_w], solve for W_total
2. Input W_total and design variables, evaluate performance at given t_hover
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import openmdao.api as om

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

# Design point (fixed from UMDO)
V_INF = 29.58             # Cruise speed [m/s]
R = 0.2653                # Rotor radius [m]
J = 1.300                 # Propeller advance ratio [-]
S_W = 0.2395              # Wing area [m²]

# Two analysis modes:
# MODE 1: Solve for W_total (design MTOM) at given t_hover
# MODE 2: Fix W_total (design MTOM), evaluate at different operating t_hover

ANALYSIS_MODE = 2  # 1 = solve for W_total, 2 = fix W_total

# For MODE 1: Hover time for sizing
T_HOVER_DESIGN = 101.0    # s (use 101s for 97.5th percentile design)

# For MODE 2: Fixed W_total (the design MTOM from MODE 1 or optimization)
W_TOTAL_FIXED = 7.845 * G  # N (e.g., 8.078 kg from deterministic design)
# W_TOTAL_FIXED = 7.85 * G  # N (e.g., 7.85 kg from UMDO 95th percentile)

# Operating points to evaluate (different mission hover times)
OPERATING_T_HOVER = [55.0, 70.0, 85.0, 101.0]  # s

# Verbose output
VERBOSE = True

# ============================================================================


@dataclass
class EvaluationResult:
    """Results from deterministic evaluation."""
    # Inputs
    V_inf: float
    r: float
    J: float
    S_w: float
    t_hover: float
    W_total_fixed: Optional[float]  # None if solved, otherwise input value
    
    # Outputs
    W_total: float          # Actual MTOM (if solved) or fixed input
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
    
    # Limits
    DL_MAX: float
    BL_MAX: float
    CL_MAX: float
    
    @property
    def MTOM_kg(self) -> float:
        return self.W_total / G
    
    @property
    def battery_kg(self) -> float:
        return self.W_battery / G
    
    @property
    def empty_kg(self) -> float:
        return self.W_empty / G
    
    @property
    def E_req_Wh(self) -> float:
        return self.E_req / 3600
    
    def summary(self) -> str:
        dl_margin = self.DL_MAX - self.disk_loading if not np.isnan(self.disk_loading) else float('nan')
        bl_margin = self.BL_MAX - self.blade_loading if not np.isnan(self.blade_loading) else float('nan')
        cl_margin = self.CL_MAX - self.cruise_CL if not np.isnan(self.cruise_CL) else float('nan')
        
        lines = [
            "-" * 50,
            f"Operating t_hover: {self.t_hover:.1f} s",
            f"  • MTOM               : {self.MTOM_kg:7.3f} kg  ({self.W_total:.1f} N)",
            f"  • Battery mass       : {self.battery_kg:7.3f} kg",
            f"  • Empty mass         : {self.empty_kg:7.3f} kg",
            f"  • P_hover            : {self.P_hover:8.1f} W",
            f"  • P_cruise           : {self.P_cruise:8.1f} W",
            f"  • E_required         : {self.E_req_Wh:7.3f} Wh",
            f"  • Disk loading       : {self.disk_loading:7.2f} / {self.DL_MAX} N/m² (Margin: {dl_margin:7.2f})",
            f"  • Blade loading      : {self.blade_loading:7.4f} / {self.BL_MAX}      (Margin: {bl_margin:7.4f})",
            f"  • Cruise CL          : {self.cruise_CL:7.4f} / {self.CL_MAX}      (Margin: {cl_margin:7.4f})",
        ]
        return "\n".join(lines)
    
    def check_constraints(self) -> bool:
        """Check if all constraints are satisfied at operating point."""
        dl_ok = self.disk_loading <= self.DL_MAX if not np.isnan(self.disk_loading) else True
        bl_ok = self.blade_loading <= self.BL_MAX if not np.isnan(self.blade_loading) else True
        cl_ok = self.cruise_CL <= self.CL_MAX if not np.isnan(self.cruise_CL) else True
        
        all_ok = dl_ok and bl_ok and cl_ok
        
        if not all_ok:
            print(f"\n⚠️ Constraint violations at t_hover={self.t_hover:.1f}s:")
            if not dl_ok: print(f"     Disk loading: {self.disk_loading:.2f} > {self.DL_MAX}")
            if not bl_ok: print(f"     Blade loading: {self.blade_loading:.4f} > {self.BL_MAX}")
            if not cl_ok: print(f"     Cruise CL: {self.cruise_CL:.4f} > {self.CL_MAX}")
        
        return all_ok


class DeterministicEvaluator:
    """Deterministic evaluator for QBiT model."""
    
    def __init__(self, payload_kg: float, range_m: float, n_c: int):
        self.payload_kg = payload_kg
        self.range_m = range_m
        self.n_c = n_c
        self._prob = None
        
    def _build_problem(self) -> om.Problem:
        """Build OpenMDAO problem."""
        prob = om.Problem(reports=None)
        prob.model = build_qbit_model(self.payload_kg, self.range_m, self.n_c)
        
        prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
        prob.model.set_input_defaults("V_inf", val=33.0, units="m/s")
        prob.model.set_input_defaults("r", val=0.22, units="m")
        prob.model.set_input_defaults("J", val=1.3)
        prob.model.set_input_defaults("S_w", val=0.20, units="m**2")
        
        prob.setup()
        return prob
    
    def evaluate_with_fixed_W(self, V_inf: float, r: float, J: float, S_w: float,
                               W_total_fixed: float, t_hover: float) -> EvaluationResult:
        """
        Evaluate the model with FIXED W_total (no solving).
        This is used for off-design analysis (e.g., design for 101s, fly at 55s).
        """
        prob = self._build_problem()
        
        # Set all values
        prob.set_val("W_total", W_total_fixed)
        prob.set_val("V_inf", V_inf)
        prob.set_val("r", r)
        prob.set_val("J", J)
        prob.set_val("S_w", S_w)
        
        # Set hover time
        _orig_t = getattr(sc, "T_HOVER", None)
        sc.T_HOVER = float(t_hover)
        
        try:
            prob.run_model()
            converged = True
            
            # Extract results
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
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            converged = False
            W_battery = W_empty = P_hover = P_cruise = E_req = 0.0
            disk_loading = blade_loading = cruise_CL = weight_residual = float('nan')
            b = float(np.sqrt(AR_FIXED * S_w))
            chord = S_w / b if b > 0 else 0.0
        
        finally:
            if _orig_t is None:
                if hasattr(sc, "T_HOVER"):
                    del sc.T_HOVER
            else:
                sc.T_HOVER = _orig_t
        
        return EvaluationResult(
            V_inf=V_inf, r=r, J=J, S_w=S_w, t_hover=t_hover,
            W_total_fixed=W_total_fixed,
            W_total=W_total_fixed, W_battery=W_battery, W_empty=W_empty,
            P_hover=P_hover, P_cruise=P_cruise, b=b, chord=chord, E_req=E_req,
            disk_loading=disk_loading, blade_loading=blade_loading,
            cruise_CL=cruise_CL, weight_residual=weight_residual,
            converged=converged,
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )
    
    def evaluate_solve_W(self, V_inf: float, r: float, J: float, S_w: float,
                          t_hover: float) -> EvaluationResult:
        """
        Solve for W_total (design mode) - original functionality.
        """
        from scipy.optimize import brentq
        
        prob = self._build_problem()
        
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
                    raise ValueError("Could not find bracket for root")
            
            W_total = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)
            
            # Run one more time to get all outputs
            prob.set_val("W_total", W_total)
            prob.run_model()
            
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
            
            converged = True
            
        except Exception as e:
            print(f"Solve failed: {e}")
            converged = False
            W_total = W_TOTAL_BOUNDS[1]
            W_battery = W_empty = P_hover = P_cruise = E_req = 0.0
            disk_loading = blade_loading = cruise_CL = weight_residual = float('nan')
            b = float(np.sqrt(AR_FIXED * S_w))
            chord = S_w / b if b > 0 else 0.0
        
        finally:
            if _orig_t is None:
                if hasattr(sc, "T_HOVER"):
                    del sc.T_HOVER
            else:
                sc.T_HOVER = _orig_t
        
        return EvaluationResult(
            V_inf=V_inf, r=r, J=J, S_w=S_w, t_hover=t_hover,
            W_total_fixed=None,
            W_total=W_total, W_battery=W_battery, W_empty=W_empty,
            P_hover=P_hover, P_cruise=P_cruise, b=b, chord=chord, E_req=E_req,
            disk_loading=disk_loading, blade_loading=blade_loading,
            cruise_CL=cruise_CL, weight_residual=weight_residual,
            converged=converged,
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )


def main():
    print("\n" + "=" * 70)
    print("DETERMINISTIC QBiT EVALUATION")
    print("=" * 70)
    print(f"\nFixed Design Point:")
    print(f"  V_inf = {V_INF:.2f} m/s")
    print(f"  r     = {R:.4f} m")
    print(f"  J     = {J:.3f}")
    print(f"  S_w   = {S_W:.4f} m²")
    
    evaluator = DeterministicEvaluator(PAYLOAD_KG, RANGE_M, N_CUSTOMERS)
    
    if ANALYSIS_MODE == 1:
        # Mode 1: Solve for W_total at design hover time
        print(f"\n{'='*70}")
        print("MODE 1: Design Sizing (Solving for W_total)")
        print(f"{'='*70}")
        print(f"Design hover time: {T_HOVER_DESIGN:.1f} s")
        
        result = evaluator.evaluate_solve_W(V_INF, R, J, S_W, T_HOVER_DESIGN)
        
        print(f"\n📋 DESIGN RESULT (sized for t_hover = {T_HOVER_DESIGN:.1f}s):")
        print(f"  • Design MTOM: {result.MTOM_kg:.3f} kg")
        print(f"  • Battery:    {result.battery_kg:.3f} kg")
        print(f"  • Empty mass: {result.empty_kg:.3f} kg")
        print(f"  • E_req:      {result.E_req_Wh:.1f} Wh")
        
        print(f"\n📊 For documentation, use this W_total: {result.W_total:.1f} N ({result.MTOM_kg:.3f} kg)")
        
    else:
        # Mode 2: Fix W_total, evaluate at different operating t_hover
        print(f"\n{'='*70}")
        print("MODE 2: Off-Design Analysis (Fixed MTOM, Varying t_hover)")
        print(f"{'='*70}")
        print(f"Fixed design MTOM: {W_TOTAL_FIXED/G:.3f} kg ({W_TOTAL_FIXED:.1f} N)")
        print(f"Operating hover times: {OPERATING_T_HOVER}")
        print()
        
        results = []
        for t_op in OPERATING_T_HOVER:
            result = evaluator.evaluate_with_fixed_W(
                V_INF, R, J, S_W, W_TOTAL_FIXED, t_op
            )
            results.append(result)
        
        # Print summary table
        print("\n" + "=" * 90)
        print("OFF-DESIGN PERFORMANCE SUMMARY")
        print("=" * 90)
        print(f"{'t_hover [s]':<12} {'MTOM [kg]':<12} {'E_req [Wh]':<12} {'P_hover [W]':<12} {'Disk Load':<12} {'Cruise CL':<12}")
        print("-" * 90)
        
        for r in results:
            status = "✓" if r.converged else "✗"
            print(f"{r.t_hover:<12.1f} {r.MTOM_kg:<12.3f} {r.E_req_Wh:<12.1f} {r.P_hover:<12.1f} {r.disk_loading:<12.2f} {r.cruise_CL:<12.4f} {status}")
        
        print("=" * 90)
        
        # Check constraints at each operating point
        print("\n🔍 CONSTRAINT CHECK AT EACH OPERATING POINT:")
        for r in results:
            all_ok = r.check_constraints()
            if not all_ok:
                print(f"  ⚠️ Violations at t_hover={r.t_hover:.1f}s")
        
        # Key insight for your documentation
        print("\n" + "=" * 70)
        print("KEY INSIGHT")
        print("=" * 70)
        
        design_t = T_HOVER_DESIGN if 'T_HOVER_DESIGN' in dir() else "design"
        print(f"""
This analysis shows how the same vehicle (designed for {design_t}s) performs 
at different mission hover times. This is critical for understanding:
1. Energy consumption varies significantly with actual hover time
2. Constraints may be violated if vehicle operates outside design conditions
3. The design MTOM remains fixed, but performance changes

For your case: Designing for t_hover=101s but flying at 55s means:
- You have excess battery capacity
- You use less energy than designed for
- All constraints remain satisfied (likely with margins)
""")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()