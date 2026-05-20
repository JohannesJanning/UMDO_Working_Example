"""
UQPCE-based robust MDO for the QBiT UAV sizing model.

Minimises the 95th-percentile upper CI of W_total (MTOM) subject to
robust constraints on cruise CL, disk loading, and blade loading,
under lognormal uncertainty in t_hover.
"""

from __future__ import annotations

import os
import time  # ← ADD THIS IMPORT
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
# Helper function for time formatting
# ---------------------------------------------------------------------------
def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes ({seconds:.0f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f} hours ({int(hours)}h {int(minutes)}m)"


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
    UQPCE uses Normal distribution (optimal Hermite polynomials).
    Transform to lognormal inside QBiTUQComp.
    """
    
    # LogNormal parameters (your original)
    shift = 25.0
    target_mean = 55.0
    target_std = 18.0
    
    # Calculate lognormal μ, σ from shifted lognormal parameters
    from scipy.optimize import fsolve
    
    def solve_lognorm_params(params):
        mu, sigma = params
        mean_ln = np.exp(mu + sigma**2/2)
        var_ln = np.exp(2*mu + sigma**2) * (np.exp(sigma**2) - 1)
        return [shift + mean_ln - target_mean, var_ln - target_std**2]
    
    mu_ln, sigma_ln = fsolve(solve_lognorm_params, [3.2, 0.5])
    
    print(f"  Lognormal parameters for transform:")
    print(f"    μ = {mu_ln:.6f}, σ = {sigma_ln:.6f}, shift = {shift}")
    print(f"    Verifying: mean = {shift + np.exp(mu_ln + sigma_ln**2/2):.2f}s (target: {target_mean}s)")
    print(f"    Verifying: std = {np.sqrt(np.exp(2*mu_ln + sigma_ln**2) * (np.exp(sigma_ln**2) - 1)):.2f}s (target: {target_std}s)")
    
    # UQPCE uses STANDARD NORMAL distribution (optimal Hermite)
    config = {
        "Variable 0": {
            "name": "z_hover",
            "distribution": "normal",
            "mean": 0.0,
            "stdev": 1.0,
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
    d["lognormal_params"] = {"mu": mu_ln, "sigma": sigma_ln, "shift": shift}
    
    return d


# ---------------------------------------------------------------------------
# Inner solver (unchanged)
# ---------------------------------------------------------------------------
def inner_solve_for_Wtotal(
    prob: om.Problem,
    t: float,
    payload: float,
    range_m: float,
    n_c: int,
    dvars: tuple[float, float, float, float],
) -> dict:
    V, r, J, Sw = dvars
    _orig_t = getattr(sc, "T_HOVER", None)
    sc.T_HOVER = float(t)

    try:
        def eval_res(W: float) -> float:
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
                raise om.AnalysisError(f"No sign change in weight residual for t={t:.1f}s")

        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)

        return {
            "W_total":      root,
            "cruise_CL":    float(prob.get_val("cruise_CL")[0]),
            "disk_loading": float(prob.get_val("disk_loading")[0]),
            "blade_loading": float(prob.get_val("blade_loading")[0]),
        }

    except Exception as exc:
        print(f"  [inner_solve] FAILED t={t:.1f}s: {exc}")
        return None

    finally:
        if _orig_t is None:
            if hasattr(sc, "T_HOVER"):
                del sc.T_HOVER
        else:
            sc.T_HOVER = _orig_t


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------
class QBiTUQComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("resp_cnt", types=int)
        self.options.declare("z_samples", types=np.ndarray)
        self.options.declare("payload_kg", types=float)
        self.options.declare("range_m", types=float)
        self.options.declare("n_c", types=int)
        self.options.declare("mu_ln", types=float, default=3.28424)
        self.options.declare("sigma_ln", types=float, default=0.55403)
        self.options.declare("shift", types=float, default=25.0)

    def setup(self):
        self.add_input("V_inf", val=33.0, units="m/s")
        self.add_input("r", val=0.22, units="m")
        self.add_input("J", val=1.3)
        self.add_input("S_w", val=0.20, units="m**2")

        n = self.options["resp_cnt"]
        self.add_output("W_total", shape=(n,), units="N")
        self.add_output("cruise_CL", shape=(n,))
        self.add_output("disk_loading", shape=(n,), units="N/m**2")
        self.add_output("blade_loading", shape=(n,))

        self.declare_partials("*", "*", method="fd", step=1e-4, step_calc="rel")

        self._inner = om.Problem(reports=None)
        inner_model = build_qbit_model(
            self.options["payload_kg"],
            self.options["range_m"],
            self.options["n_c"],
        )
        self._inner.model = inner_model
        inner_model.set_input_defaults("W_total", val=6.0 * G, units="N")
        inner_model.set_input_defaults("V_inf", val=33.0, units="m/s")
        inner_model.set_input_defaults("r", val=0.22, units="m")
        inner_model.set_input_defaults("J", val=1.3)
        inner_model.set_input_defaults("S_w", val=0.20, units="m**2")

        self._inner.setup()
        self._inner.run_model()

    def compute(self, inputs, outputs):
        dvars = (
            float(inputs["V_inf"][0]),
            float(inputs["r"][0]),
            float(inputs["J"][0]),
            float(inputs["S_w"][0]),
        )
        
        z_samples = self.options["z_samples"]
        mu_ln = self.options["mu_ln"]
        sigma_ln = self.options["sigma_ln"]
        shift = self.options["shift"]
        
        t_samples = shift + np.exp(mu_ln + sigma_ln * z_samples)
        
        pl = self.options["payload_kg"]
        rm = self.options["range_m"]
        nc = self.options["n_c"]

        W_arr = np.empty(len(t_samples))
        cl_arr = np.empty(len(t_samples))
        dl_arr = np.empty(len(t_samples))
        bl_arr = np.empty(len(t_samples))

        W_PENALTY = float(W_TOTAL_BOUNDS[1])
        CL_PENALTY = CL_MAX * 1.5
        DL_PENALTY = DL_MAX * 1.5
        BL_PENALTY = BL_MAX * 1.5

        n_failed = 0
        for i, t in enumerate(t_samples):
            res = inner_solve_for_Wtotal(self._inner, t, pl, rm, nc, dvars)
            if res is None:
                n_failed += 1
                W_arr[i] = W_PENALTY
                cl_arr[i] = CL_PENALTY
                dl_arr[i] = DL_PENALTY
                bl_arr[i] = BL_PENALTY
            else:
                W_arr[i] = res["W_total"]
                cl_arr[i] = res["cruise_CL"]
                dl_arr[i] = res["disk_loading"]
                bl_arr[i] = res["blade_loading"]

        if n_failed:
            print(f"  [compute] {n_failed}/{len(t_samples)} samples failed")

        outputs["W_total"] = W_arr
        outputs["cruise_CL"] = cl_arr
        outputs["disk_loading"] = dl_arr
        outputs["blade_loading"] = bl_arr


# ---------------------------------------------------------------------------
# Main optimisation
# ---------------------------------------------------------------------------
def run():
    # Record total start time
    total_start_time = time.time()
    
    print("=" * 70)
    print("UQPCE ROBUST DESIGN OPTIMIZATION")
    print("=" * 70)
    
    # ---- 1. UQPCE initialisation ----------------------------------------
    print("\n[1/5] Setting up UQPCE...")
    setup_start = time.time()
    d = setup_and_init_uqpce()
    n_quad = d["resp_cnt"]
    z_samples = d["run_matrix"][:, 0]
    lognormal_params = d["lognormal_params"]
    setup_time = time.time() - setup_start
    print(f"      ✓ Setup completed in {format_time(setup_time)}")
    print(f"      • PCE quadrature points: {n_quad}")
    print(f"      • Transform: t_hover = {lognormal_params['shift']} + exp({lognormal_params['mu']:.4f} + {lognormal_params['sigma']:.4f}·Z)")
    
    # ---- 2. Build outer problem -----------------------------------------
    print("\n[2/5] Building optimization problem...")
    prob = om.Problem(reports=None)
    
    ivc = om.IndepVarComp()
    ivc.add_output("V_inf", val=35.0, units="m/s")
    ivc.add_output("r", val=0.20, units="m")
    ivc.add_output("J", val=1.3)
    ivc.add_output("S_w", val=0.30, units="m**2")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    
    comp = QBiTUQComp(
        resp_cnt=n_quad,
        z_samples=z_samples,
        payload_kg=3.0,
        range_m=15000.0,
        n_c=2,
        mu_ln=lognormal_params["mu"],
        sigma_ln=lognormal_params["sigma"],
        shift=lognormal_params["shift"],
    )
    prob.model.add_subsystem("eval", comp, promotes=["*"])
    
    uq = UQPCEGroup(
        significance=d["significance"],
        var_basis=d["var_basis"],
        norm_sq=d["norm_sq"],
        resampled_var_basis=d["resampled_var_basis"],
        tail="upper",
        epistemic_cnt=d["epistemic_cnt"],
        aleatory_cnt=d["aleatory_cnt"],
        uncert_list=["W_total", "cruise_CL", "disk_loading", "blade_loading"],
        tanh_omega=0.05,
        sample_ref0=[63.16, 0.494, 76.5, 0.0130],
        sample_ref=[2.53, 0.018, 5.5, 0.0002],
    )
    promoted = [
        "W_total", "cruise_CL", "disk_loading", "blade_loading",
        "W_total:ci_upper", "W_total:mean", "W_total:variance",
        "cruise_CL:ci_upper",
        "disk_loading:ci_upper",
        "blade_loading:ci_upper",
    ]
    prob.model.add_subsystem("uq", uq, promotes=promoted)
    
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=50, tol=1e-4)
    prob.driver.options["debug_print"] = ["objs", "nl_cons", "desvars"]
    
    prob.model.add_design_var("V_inf", lower=V_INF_BOUNDS[0], upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r", lower=R_BOUNDS[0], upper=R_BOUNDS[1])
    prob.model.add_design_var("J", lower=J_BOUNDS[0], upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w", lower=S_W_BOUNDS[0], upper=S_W_BOUNDS[1])
    
    prob.model.add_constraint("cruise_CL:ci_upper", upper=CL_MAX)
    prob.model.add_constraint("disk_loading:ci_upper", upper=DL_MAX)
    prob.model.add_constraint("blade_loading:ci_upper", upper=BL_MAX)
    
    prob.model.add_objective("W_total:ci_upper")
    
    prob.setup()
    print("      ✓ Problem built successfully")
    
    # ---- 3. PCE validation at initial design ----------------------------
    print("\n[3/5] Validating PCE surrogate at initial design...")
    prob.run_model()
    W_quad = prob.get_val("W_total")
    pce_mean = prob.get_val("W_total:mean")[0]
    pce_var = prob.get_val("W_total:variance")[0]
    pce_ci = prob.get_val("W_total:ci_upper")[0]
    G_val = 9.80665
    print(f"      • PCE mean: {pce_mean/G_val:.3f} kg")
    print(f"      • PCE std:  {pce_var**0.5:.2f} N")
    print(f"      • PCE 95th: {pce_ci/G_val:.3f} kg")
    
    # ---- 4. Run optimisation --------------------------------------------
    print("\n[4/5] Running robust optimization...")
    print("-" * 50)
    opt_start_time = time.time()
    prob.run_driver()
    opt_end_time = time.time()
    opt_duration = opt_end_time - opt_start_time
    print("-" * 50)
    
    # ---- 5. Results -----------------------------------------------------
    success = prob.driver.result.success if hasattr(prob.driver, "result") else "N/A"
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
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
    
    # Get iteration info
    if hasattr(prob.driver, "iter_count"):
        print(f"\n  Optimization iterations: {prob.driver.iter_count}")
    if hasattr(prob.driver, "result") and hasattr(prob.driver.result, "nfev"):
        print(f"  Function evaluations: {prob.driver.result.nfev}")
    
    print("=" * 60)
    
    # ---- Validation at optimal design -----------------------------------
    print("\n[5/5] Validating at optimal design with Monte Carlo...")
    val_start_time = time.time()
    
    from run_qbit_MCS import RobustOptimizer, sample_t_hover
    uq_ref = RobustOptimizer(payload_kg=3.0, range_m=15000.0, n_c=2, n_mc=10000, seed=42)
    uq_ref.mc_samples = sample_t_hover(2000, uq_ref.mean_t, uq_ref.std_t,
                                        uq_ref.shift_t, seed=42)
    x0 = [prob.get_val("V_inf")[0], prob.get_val("r")[0],
          prob.get_val("J")[0], prob.get_val("S_w")[0]]
    mc_stats = uq_ref._mc_stats(x0)
    
    val_duration = time.time() - val_start_time
    
    if mc_stats:
        mc_W = [r["W_total"] for r in mc_stats["results"] if r]
        mc_mean = np.mean(mc_W)
        mc_std = np.std(mc_W)
        pce_mean = prob.get_val("W_total:mean")[0]
        pce_std = prob.get_val("W_total:variance")[0]**0.5
        print(f"\n  PCE mean = {pce_mean:.2f} N,  MC mean = {mc_mean:.2f} N,  "
              f"error = {abs(pce_mean-mc_mean):.2f} N ({abs(pce_mean-mc_mean)/mc_mean*100:.1f}%)")
        print(f"  PCE std  = {pce_std:.2f} N,   MC std  = {mc_std:.2f} N,   "
              f"error = {abs(pce_std-mc_std):.2f} N ({abs(pce_std-mc_std)/mc_std*100:.1f}%)")
    
    # ---- TIMING SUMMARY -------------------------------------------------
    total_duration = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"PCE setup time     : {format_time(setup_time)}")
    print(f"Optimization time  : {format_time(opt_duration)}")
    print(f"Validation time    : {format_time(val_duration)}")
    print(f"Total runtime      : {format_time(total_duration)}")
    print("=" * 60)
    print("\n OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()