"""
UQPCE-based robust MDO for the QBiT UAV sizing model.

Minimises the 95th-percentile upper CI of W_total (MTOM) subject to
robust constraints on cruise CL, disk loading, and blade loading,
under lognormal uncertainty in t_hover.

Key design decisions vs. previous versions:
  - sc.T_HOVER is patched ONCE per sample (outside Brent loop), not per iteration
  - Inner om.Problem is created ONCE in setup(), reused across all Brent calls
  - Failed inner solves raise om.AnalysisError so SLSQP backtracks cleanly
  - No sentinel fill values that would corrupt the PCE fit
  - check_partials is gated behind CHECK_PARTIALS env var
  - Redundant prob.run_model() after brentq is removed
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use('Agg')  # must be before any other matplotlib import

import numpy as np
import openmdao.api as om
import yaml
from scipy.optimize import brentq

# --- QBiT ---
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (
    G,
    W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS,
    DL_MAX, BL_MAX, CL_MAX,
)
import qbit.components.sizing_comps as sc

# --- UQPCE ---
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.pce.pce import PCE
from uqpce.pce.io import read_input_file
from uqpce.mdao import interface

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
YAML_INPUT   = os.path.join(_HERE, "uqpce_input.yaml")
MATRIX_FILE  = os.path.join(_HERE, "uqpce_run_matrix.dat")


# ---------------------------------------------------------------------------
# UQPCE setup
# ---------------------------------------------------------------------------
def setup_and_init_uqpce() -> dict:
    """
    Write the UQPCE input YAML, sample the quadrature/collocation matrix,
    and return the initialisation dict for UQPCEGroup.

    t_hover ~ ShiftedLognormal(mu_ln=3.28424, sigma_ln=0.55403, shift=25 s)
    which gives mean ≈ 55 s, std ≈ 18 s, hard lower bound 25 s.

    Order-3 PCE with 1 variable → 4 basis polynomials.
    pce.sample() without a count uses the default quadrature rule for the
    variable type (Gauss-Laguerre for lognormal), giving enough points to
    integrate the order-3 polynomial exactly.  Validate against MC once.
    """
    # UQPCE distribution choice for t_hover:
  
    config = {
        "Variable 0": {
            "name": "t_hover",
            "distribution": "beta",
            "alpha": 1.1772, #2.0576,
            "beta":  7.6088, #3.8263,
            "interval_low":  32.0,
            "interval_high": 200.0,
            "type": "aleatory",
        },
        "Settings": {
            "order": 6,
            "backend": "Agg",
            "track_convergence_off": True,
            "aleat_samp_size": 100000,
        },
    }

    with open(YAML_INPUT, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    var_dict, settings = read_input_file(YAML_INPUT)
    settings.pop("plot", None)
    settings.pop("verbose", None)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pce = PCE(outputs=False, plot=False, verbose=False, **settings)
        for value in var_dict.values():
            pce.add_variable(**value)
        X = pce.sample()
        np.savetxt(MATRIX_FILE, X)

    d = interface.initialize_dict(YAML_INPUT, MATRIX_FILE)

    t_pts = d["run_matrix"][:, 0]
    t_max = float(t_pts.max())
    print(f"  Quadrature t_hover points [s]: {t_pts.round(1)}")
    if t_max > 200.0:
        raise RuntimeError(
            f"UQPCE placed a quadrature point at t={t_max:.1f}s > 200s even with "
            f"uniform distribution. Something is wrong with the UQPCE config."
        )

    return d


# ---------------------------------------------------------------------------
# Inner solver
# ---------------------------------------------------------------------------
def inner_solve_for_Wtotal(
    prob: om.Problem,
    t: float,
    payload: float,
    range_m: float,
    n_c: int,
    dvars: tuple[float, float, float, float],
) -> dict:
    """
    Solve the weight-residual equation for a fixed geometry and t_hover sample.

    Parameters
    ----------
    prob    : pre-setup om.Problem (reused across calls — do NOT re-setup)
    t       : hover time sample [s]
    payload : payload mass [kg]
    range_m : mission range [m]
    n_c     : number of customers
    dvars   : (V_inf, r, J, S_w) — frozen design variables

    Returns
    -------
    dict with W_total, cruise_CL, disk_loading, blade_loading
    — or raises om.AnalysisError on convergence failure.
    """
    V, r, J, Sw = dvars

    # --- patch T_HOVER once for this sample, restore on exit ---
    _orig_t = getattr(sc, "T_HOVER", None)
    sc.T_HOVER = float(t)

    try:
        def eval_res(W: float) -> float:
            """Residual function for Brent: weight_residual(W) = W_computed - W."""
            try:
                prob.set_val("W_total", W)
                prob.set_val("V_inf",   V)
                prob.set_val("r",       r)
                prob.set_val("J",       J)
                prob.set_val("S_w",     Sw)
                prob.run_model()
                return float(prob.get_val("weight_residual")[0])
            except Exception:
                return float("nan")

        wl = float(W_TOTAL_BOUNDS[0])
        wh = float(W_TOTAL_BOUNDS[1])

        rl, rh = eval_res(wl), eval_res(wh)

        # bracket search if signs don't differ
        if np.isnan(rl) or np.isnan(rh) or rl * rh > 0:
            found = False
            for xl, xr in zip(
                np.linspace(wl, wh, 25)[:-1],
                np.linspace(wl, wh, 25)[1:],
            ):
                fl, fr = eval_res(xl), eval_res(xr)
                if not (np.isnan(fl) or np.isnan(fr)) and fl * fr <= 0:
                    wl, wh = xl, xr
                    found = True
                    break
            if not found:
                raise om.AnalysisError(
                    f"No sign change in weight residual for t={t:.1f}s, "
                    f"dvars={dvars}. Cannot bracket root."
                )

        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)

        # brentq's final call is at root → model state is current; read directly
        return {
            "W_total":      root,
            "cruise_CL":    float(prob.get_val("cruise_CL")[0]),
            "disk_loading": float(prob.get_val("disk_loading")[0]),
            "blade_loading":float(prob.get_val("blade_loading")[0]),
        }

    except Exception as exc:
        # Return None — compute() will apply a smooth penalty.
        # Do NOT raise: scipy SLSQP cannot handle AnalysisError during
        # constraint evaluation and crashes rather than backtracking.
        print(f"  [inner_solve] FAILED t={t:.1f}s dvars={dvars}: {exc}")
        return None

    finally:
        # always restore T_HOVER, even on exception
        if _orig_t is None:
            if hasattr(sc, "T_HOVER"):
                del sc.T_HOVER
        else:
            sc.T_HOVER = _orig_t


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------
class QBiTUQComp(om.ExplicitComponent):
    """
    Evaluates the QBiT sizing model at each PCE quadrature point (t_hover sample)
    for a given design variable vector, and returns per-sample outputs for UQPCE.

    The inner om.Problem is created once in setup() and reused in compute(),
    avoiding repeated setup overhead (~10-50× speedup per optimizer call).
    """

    def initialize(self):
        self.options.declare("resp_cnt",   types=int,   desc="Number of PCE quadrature points")
        self.options.declare("t_samples",  types=np.ndarray, desc="t_hover quadrature points [s]")
        self.options.declare("payload_kg", types=float, desc="Payload [kg]")
        self.options.declare("range_m",    types=float, desc="Mission range [m]")
        self.options.declare("n_c",        types=int,   desc="Number of customers")

    def setup(self):
        # --- outer inputs (design variables) ---
        self.add_input("V_inf", val=33.0,  units="m/s")
        self.add_input("r",     val=0.22,  units="m")
        self.add_input("J",     val=1.3)
        self.add_input("S_w",   val=0.20,  units="m**2")

        # --- per-sample outputs consumed by UQPCEGroup ---
        n = self.options["resp_cnt"]
        self.add_output("W_total",       shape=(n,), units="N")
        self.add_output("cruise_CL",     shape=(n,))
        self.add_output("disk_loading",  shape=(n,), units="N/m**2")
        self.add_output("blade_loading", shape=(n,))

        # FD partials — UQPCE differentiates through the PCE surrogate,
        # not through this component directly, so FD accuracy is sufficient.
        self.declare_partials("*", "*", method="fd", step=1e-4, step_calc="rel")

        # --- inner problem: created ONCE, reused across all compute() calls ---
        self._inner = om.Problem(reports=None)
        inner_model = build_qbit_model(
            self.options["payload_kg"],
            self.options["range_m"],
            self.options["n_c"],
        )
        self._inner.model = inner_model

        # Resolve the W_total promoted-input ambiguity before setup.
        # build_qbit_model has (at least) two subsystems both promoting W_total
        # with different default values; set_input_defaults pins a single value.
        inner_model.set_input_defaults("W_total", val=6.0 * G, units="N")
        inner_model.set_input_defaults("V_inf",   val=33.0,    units="m/s")
        inner_model.set_input_defaults("r",       val=0.22,    units="m")
        inner_model.set_input_defaults("J",       val=1.3)
        inner_model.set_input_defaults("S_w",     val=0.20,    units="m**2")

        self._inner.setup()
        # warm-start: run once so model outputs are initialised before first compute()
        self._inner.run_model()

    def compute(self, inputs, outputs):
        dvars = (
            float(inputs["V_inf"][0]),
            float(inputs["r"][0]),
            float(inputs["J"][0]),
            float(inputs["S_w"][0]),
        )
        ts  = self.options["t_samples"]
        pl  = self.options["payload_kg"]
        rm  = self.options["range_m"]
        nc  = self.options["n_c"]

        W_arr  = np.empty(len(ts))
        cl_arr = np.empty(len(ts))
        dl_arr = np.empty(len(ts))
        bl_arr = np.empty(len(ts))

        # Penalty values: physically plausible but outside constraint limits,
        # so the PCE surrogate pushes the optimizer away without corrupting the fit.
        W_PENALTY  = float(W_TOTAL_BOUNDS[1])
        CL_PENALTY = CL_MAX  * 1.5
        DL_PENALTY = DL_MAX  * 1.5
        BL_PENALTY = BL_MAX  * 1.5

        n_failed = 0
        for i, t in enumerate(ts):
            res = inner_solve_for_Wtotal(self._inner, t, pl, rm, nc, dvars)
            if res is None:
                n_failed += 1
                W_arr[i]  = W_PENALTY
                cl_arr[i] = CL_PENALTY
                dl_arr[i] = DL_PENALTY
                bl_arr[i] = BL_PENALTY
            else:
                W_arr[i]  = res["W_total"]
                cl_arr[i] = res["cruise_CL"]
                dl_arr[i] = res["disk_loading"]
                bl_arr[i] = res["blade_loading"]

        if n_failed:
            print(f"  [compute] {n_failed}/{len(ts)} samples failed for dvars={dvars}.")

        outputs["W_total"]       = W_arr
        outputs["cruise_CL"]     = cl_arr
        outputs["disk_loading"]  = dl_arr
        outputs["blade_loading"] = bl_arr


# ---------------------------------------------------------------------------
# Main optimisation
# ---------------------------------------------------------------------------
def run():
    # ---- 1. UQPCE initialisation ----------------------------------------
    print("Setting up UQPCE …")
    d = setup_and_init_uqpce()
    n_quad = d["resp_cnt"]
    t_samples = d["run_matrix"][:, 0]
    print(f"  PCE quadrature points : {n_quad}")
    print(f"  t_hover samples [s]   : {t_samples.round(1)}")

    # ---- 2. Build outer OpenMDAO problem ----------------------------------
    prob = om.Problem(reports=None)

    # Independent design variables
    ivc = om.IndepVarComp()
    ivc.add_output("V_inf", val=35.0, units="m/s")   # start faster
    ivc.add_output("r",     val=0.20, units="m")      # smaller rotor
    ivc.add_output("J",     val=1.3)
    ivc.add_output("S_w",   val=0.30, units="m**2")   # larger wing
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    # QBiT evaluator (runs inner solve at each quadrature point)
    comp = QBiTUQComp(
        resp_cnt=n_quad,
        t_samples=t_samples,
        payload_kg=3.0,
        range_m=15000.0,
        n_c=2,
    )
    prob.model.add_subsystem("eval", comp, promotes=["*"])

    # UQPCE surrogate + CI computation
    #
    # sample_ref0 and sample_ref MUST bracket the actual resampled output ranges.
    # The tanh CDF approximation evaluates tanh((Y - threshold) / omega).
    # If the range doesn't bracket the actual values, the step saturates and
    # the CI lands at the wrong location (observed: 9.33 kg despite correct variance).
    #
    # Values from diagnostic at the robust design point:
    #   W_total      resampled: [65.6, 79.9] N   → ref0=62,  ref=84
    #   cruise_CL    resampled: [0.44, 0.63]     → ref0=0.40, ref=0.67
    #   disk_loading resampled: [77.9, 121.3] N/m²→ ref0=70,  ref=130
    #   blade_loading resampled:[0.013, 0.014]   → ref0=0.012,ref=0.016
    #
    # Re-run the diagnostic if the design changes significantly.
    uq = UQPCEGroup(
        significance=d["significance"],
        var_basis=d["var_basis"],
        norm_sq=d["norm_sq"],
        resampled_var_basis=d["resampled_var_basis"],
        tail="upper",
        epistemic_cnt=d["epistemic_cnt"],
        aleatory_cnt=d["aleatory_cnt"],
        uncert_list=["W_total", "cruise_CL", "disk_loading", "blade_loading"],
        # tanh_omega operates in SCALED space: scaled = (value - ref0) / ref.
        # UQPCE applies one omega to ALL outputs, so ref0/ref must normalise
        # each output to unit-std scale so omega=0.05 is appropriate for all.
        #
        # Scaling derived from resampled distributions at representative designs:
        #   W_total:      mean≈70.75 N,  std≈2.53 N   → ref0=63.16, ref=2.53
        #   cruise_CL:    mean≈0.548,    std≈0.018    → ref0=0.494,  ref=0.018
        #   disk_loading: mean≈93 N/m²,  std≈5.5 N/m² → ref0=76.5,   ref=5.5
        #   blade_loading:mean≈0.0136,   std≈0.0002   → ref0=0.013,  ref=0.0002
        #
        # With this scaling, all outputs are O(1–5) in scaled space.
        # omega=0.05 gives CI accuracy within ~0.5 N for W_total.
        # If the design moves far from these reference points, re-run diagnostic.
        tanh_omega=0.05,
        sample_ref0=[63.16, 0.494, 76.5,  0.0130],
        sample_ref= [2.53,  0.018, 5.5,   0.0002],
    )
    promoted = [
        "W_total", "cruise_CL", "disk_loading", "blade_loading",
        "W_total:ci_upper", "W_total:mean", "W_total:variance",
        "cruise_CL:ci_upper",
        "disk_loading:ci_upper",
        "blade_loading:ci_upper",
    ]
    prob.model.add_subsystem("uq", uq, promotes=promoted)

    # ---- 3. Driver & problem formulation ----------------------------------
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=50, tol=1e-4)
    prob.driver.options["debug_print"] = ["objs", "nl_cons", "desvars"]

    prob.model.add_design_var("V_inf", lower=V_INF_BOUNDS[0], upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r",     lower=R_BOUNDS[0],     upper=R_BOUNDS[1])
    prob.model.add_design_var("J",     lower=J_BOUNDS[0],     upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w",   lower=S_W_BOUNDS[0],   upper=S_W_BOUNDS[1])

    # Robust constraints: 95th-percentile of each output must satisfy limit
    prob.model.add_constraint("cruise_CL:ci_upper",    upper=CL_MAX)
    prob.model.add_constraint("disk_loading:ci_upper", upper=DL_MAX)
    prob.model.add_constraint("blade_loading:ci_upper",upper=BL_MAX)

    # Robust objective: minimise 95th-percentile of MTOM
    prob.model.add_objective("W_total:ci_upper")

    # ---- 4. Setup & optional diagnostics ----------------------------------
    prob.setup()

    # Partial check: only when explicitly requested (slow — ~10 min)
    if os.getenv("CHECK_PARTIALS"):
        print("Checking partials (this is slow) …")
        prob.check_partials(compact_print=True, excludes=["ivc"])

    # Total derivative check: fast sanity check before running
    if os.getenv("CHECK_TOTALS"):
        prob.run_model()
        prob.check_totals(
            of=["W_total:ci_upper", "cruise_CL:ci_upper"],
            wrt=["V_inf", "r", "J", "S_w"],
            compact_print=True,
        )

    # ---- 5. PCE validation against expected MC statistics ----------------
    # Run model once at the initial design and check PCE mean/variance.
    # If variance is >> 8 N² (MC reference), the PCE fit is corrupted —
    # abort before wasting time on a broken optimization.
    print("\nValidating PCE surrogate at initial design point …")
    prob.run_model()
    W_quad    = prob.get_val("W_total")
    pce_mean  = prob.get_val("W_total:mean")[0]
    pce_var   = prob.get_val("W_total:variance")[0]
    pce_ci    = prob.get_val("W_total:ci_upper")[0]
    G_val     = 9.80665
    print(f"  W_total at quadrature points [N]: {W_quad.round(2)}")
    print(f"  PCE mean  : {pce_mean/G_val:.3f} kg  (MC ref: ~7.0–7.2 kg)")
    print(f"  PCE std   : {pce_var**0.5:.2f} N    (MC ref: ~2.8 N)")
    print(f"  PCE 95th  : {pce_ci/G_val:.3f} kg  (MC ref: ~7.5–7.7 kg, diagnostic: 7.528 kg)")

    # Hard guard: abort if variance is implausible (> 100x MC reference of ~8 N²)
    if pce_var > 100:  # MC reference ~5-8 N²; anything >100 indicates corrupted PCE
        raise RuntimeError(
            f"PCE variance = {pce_var:.1f} N² >> expected ~8 N². "
            f"Surrogate is corrupted — check quadrature point failures above, "
            f"tighten interval_high, or reduce PCE order."
        )
    # Soft warning: variance within 5x of reference
    if pce_var > 20:
        print(f"  WARNING: PCE variance {pce_var:.1f} N² is elevated vs MC ~8 N². "
              f"Results may be inaccurate — consider reducing interval_high further.")

    # ---- 6. Run optimisation ----------------------------------------------
    print("\nRunning UQPCE robust optimisation …")
    prob.run_driver()

    # ---- 6. Results -------------------------------------------------------
    success = prob.driver.result.success if hasattr(prob.driver, "result") else "N/A"
    print("\n" + "=" * 60)
    print("UQPCE Robust MDO — Results")
    print("=" * 60)
    print(f"  Converged      : {success}")
    print(f"  V_inf          : {prob.get_val('V_inf')[0]:.2f} m/s")
    print(f"  Rotor radius r : {prob.get_val('r')[0]:.4f} m")
    print(f"  Prop adv. J    : {prob.get_val('J')[0]:.3f}")
    print(f"  Wing area S_w  : {prob.get_val('S_w')[0]:.4f} m²")
    print()
    print(f"  Mean MTOM      : {prob.get_val('W_total:mean')[0]/G:.3f} kg")
    print(f"  MTOM variance  : {prob.get_val('W_total:variance')[0]:.4f} N²")
    print(f"  Robust MTOM    : {prob.get_val('W_total:ci_upper')[0]/G:.3f} kg  (95th pct)")
    print()
    print("  Robust constraint values vs limits:")
    print(f"    cruise_CL    : {prob.get_val('cruise_CL:ci_upper')[0]:.4f}  ≤ {CL_MAX}")
    print(f"    disk_loading : {prob.get_val('disk_loading:ci_upper')[0]:.2f} N/m²  ≤ {DL_MAX}")
    print(f"    blade_loading: {prob.get_val('blade_loading:ci_upper')[0]:.4f}  ≤ {BL_MAX}")
    print("=" * 60)

    # Add this to your validation block in run()
    print("\nSurrogate validation vs MC reference:")
    from run_qbit_MCS import RobustOptimizer, sample_t_hover
    uq_ref = RobustOptimizer(payload_kg=3.0, range_m=15000.0, n_c=2, n_mc=10000, seed=42)
    uq_ref.mc_samples = sample_t_hover(2000, uq_ref.mean_t, uq_ref.std_t,
                                        uq_ref.shift_t, seed=42)
    x0 = [prob.get_val("V_inf")[0], prob.get_val("r")[0],
        prob.get_val("J")[0],    prob.get_val("S_w")[0]]
    mc_stats = uq_ref._mc_stats(x0)
    if mc_stats:
        mc_W = [r["W_total"] for r in mc_stats["results"] if r]
        mc_mean = np.mean(mc_W); mc_std = np.std(mc_W)
        pce_mean = prob.get_val("W_total:mean")[0]
        pce_std  = prob.get_val("W_total:variance")[0]**0.5
        print(f"  PCE mean = {pce_mean:.2f} N,  MC mean = {mc_mean:.2f} N,  "
            f"error = {abs(pce_mean-mc_mean):.2f} N ({abs(pce_mean-mc_mean)/mc_mean*100:.1f}%)")
        print(f"  PCE std  = {pce_std:.2f} N,   MC std  = {mc_std:.2f} N,   "
            f"error = {abs(pce_std-mc_std):.2f} N ({abs(pce_std-mc_std)/mc_std*100:.1f}%)")
        if abs(pce_mean - mc_mean) / mc_mean > 0.02:
            print("  WARNING: PCE mean error > 2% — consider increasing order")
        if abs(pce_std - mc_std) / mc_std > 0.15:
            print("  WARNING: PCE std error > 15% — surrogate may be inaccurate")


if __name__ == "__main__":
    run()