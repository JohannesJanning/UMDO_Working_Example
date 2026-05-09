"""
run_hexarotor.py - Hexarotor Stage-1 sizing optimization driver.

Minimise W_total subject to:
  - weight_residual = 0         Eq. (2), weight closure
  - disk_loading   <= 250 N/m²  Table 1
  - blade_loading  <= 0.14      Table 1
  (No CL constraint — QBiT only per Table 1)

Design variables: W_total, V_inf, r, mu  [Table 1 bounds]
  (No J, no S_w — hexarotor has no wing and uses μ not J)

Usage:
    python run_hexarotor.py
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import openmdao.api as om
om.config_reports = False

from hexarotor.models.hexarotor_model import build_hexarotor_model
from hexarotor.constants import (
    BATTERY_EFF, G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS, MU_BOUNDS,
    DL_MAX, BL_MAX,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SizingResult:
    W_total:   float
    W_battery: float
    W_empty:   float
    P_hover:   float
    P_cruise:  float
    V_inf:     float
    r:         float
    mu:        float
    beta_deg:  float
    E_req:     float
    converged: bool
    disk_loading: float
    blade_loading: float
    weight_residual: float
    DL_MAX: float
    BL_MAX: float

    def summary(self) -> str:
        dl_margin = self.DL_MAX - self.disk_loading
        bl_margin = self.BL_MAX - self.blade_loading
        lines = [
            f"  MTOM          : {self.W_total / G:7.3f} kg  ({self.W_total:.1f} N)",
            f"  Battery mass  : {self.W_battery / G:7.3f} kg",
            f"  Empty mass    : {self.W_empty / G:7.3f} kg",
            f"  Cruise speed  : {self.V_inf:7.2f} m/s",
            f"  Rotor radius  : {self.r:7.4f} m",
            f"  Adv. ratio μ  : {self.mu:7.4f}",
            f"  Shaft tilt β  : {self.beta_deg:7.2f}°",
            f"  P_hover       : {self.P_hover:8.1f} W",
            f"  P_cruise      : {self.P_cruise:8.1f} W",
            f"  E_required    : {self.E_req / 3600:.3f} Wh",
            f"  Converged     : {self.converged}",
            "--- Constraints & Margins ---",
            f"  Weight Resid. : {self.weight_residual:10.4e} (Goal: 0.0)",
            f"  Disk Loading  : {self.disk_loading:7.2f} / {self.DL_MAX} N/m² (Margin: {dl_margin:7.2f})",
            f"  Blade Loading : {self.blade_loading:7.4f} / {self.BL_MAX}      (Margin: {bl_margin:7.4f})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build problem
# ---------------------------------------------------------------------------
def build_problem(payload_kg: float,
                  range_m:    float,
                  n_c:        int = 1) -> om.Problem:
    prob = om.Problem(reports=None)
    prob.model = build_hexarotor_model(payload_kg, range_m, n_c)

    # Fix ambiguous initial value for W_total (not from ivc, only from balance comp)
    prob.model.set_input_defaults('W_total', val=5.0 * G, units='N')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol']       = 1e-9
    prob.driver.options['maxiter']   = 2000

    # Design variables (Table 1)
    prob.model.add_design_var('W_total', lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var('V_inf',   lower=V_INF_BOUNDS[0],   upper=V_INF_BOUNDS[1])
    prob.model.add_design_var('r',       lower=R_BOUNDS[0],       upper=R_BOUNDS[1])
    prob.model.add_design_var('mu',      lower=MU_BOUNDS[0],      upper=MU_BOUNDS[1])

    prob.model.add_objective('W_total')

    # Weight closure equality + aerodynamic constraints
    prob.model.add_constraint('weight_residual', equals=0.0)
    prob.model.add_constraint('disk_loading',    upper=DL_MAX)
    prob.model.add_constraint('blade_loading',   upper=BL_MAX)
    # No CL constraint for hexarotor

    prob.setup()
    prob.set_val('W_total', 5.0 * G)
    return prob


# ---------------------------------------------------------------------------
# Stage 1 wrapper
# ---------------------------------------------------------------------------
class Stage1Problem:
    """
    Convenience wrapper:
        Stage1Problem(payload_kg=3.0, range_m=15_000).solve(verbose=True)
    """

    def __init__(self, payload_kg: float, range_m: float, n_c: int = 1):
        self.payload_kg = payload_kg
        self.range_m    = range_m
        self.n_c          = n_c

    def solve(self, verbose: bool = True) -> SizingResult:
        prob = build_problem(self.payload_kg, self.range_m, self.n_c)

        if verbose:
            print("=" * 60)
            print("Hexarotor OpenMDAO Model")
            print(f"Single UAV - {self.n_c} Node(s) - {self.range_m*2/1000:.1f} km Range - Minimise MTOM")
            print("=" * 60)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_driver()

        W    = float(prob.get_val('W_total')[0])
        Ph   = float(prob.get_val('P_hover')[0] / BATTERY_EFF)
        Pc   = float(prob.get_val('P_cruise')[0] / BATTERY_EFF)
        V    = float(prob.get_val('V_inf')[0])
        r    = float(prob.get_val('r')[0])
        mu   = float(prob.get_val('mu')[0])
        beta = float(prob.get_val('beta')[0])
        Wb   = float(prob.get_val('W_battery')[0])
        We   = float(prob.get_val('W_empty')[0])
        E    = float(prob.get_val('E_req')[0])
        dl = float(prob.get_val('disk_loading')[0])
        bl = float(prob.get_val('blade_loading')[0])
        wr = float(prob.get_val('weight_residual')[0])

        converged = getattr(getattr(prob.driver, 'result', None), 'success', True)

        result = SizingResult(
            W_total=W, W_battery=Wb, W_empty=We,
            P_hover=Ph, P_cruise=Pc,
            V_inf=V, r=r, mu=mu,
            beta_deg=np.degrees(beta),
            E_req=E, converged=converged,
            disk_loading=dl, blade_loading=bl, weight_residual=wr,
            DL_MAX=DL_MAX, BL_MAX=BL_MAX
        )

        if verbose:
            print(result.summary())
            print("=" * 60)

        return result


if __name__ == '__main__':
    res = Stage1Problem(payload_kg=3.0, range_m=15_000.0, n_c=3).solve(verbose=True)
    mtom_kg = res.W_total / G
    assert 0.5 <= mtom_kg <= 50.0, f"MTOM {mtom_kg:.3f} kg outside [0.5, 50.0]"
    assert res.converged
    print(f"\nValidation passed: MTOM = {mtom_kg:.3f} kg ✓")
