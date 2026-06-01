"""
UQPCE-based robust MDO for the QBiT UAV sizing model.

Minimises the 95th-percentile upper CI of W_total (MTOM) subject to
robust constraints on cruise CL, disk loading, and blade loading,
under joint uncertainty in:
  1. t_hover (operational, lognormal)
  2. eta_hover (epistemic, truncated normal)
"""

from __future__ import annotations

import os
import time
import warnings
from scipy.stats import norm, truncnorm
import matplotlib
matplotlib.use('Agg')

import numpy as np
import openmdao.api as om
import yaml
from scipy.optimize import brentq
from scipy.stats import truncnorm

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
# Helper functions
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


def get_shifted_lognormal_params(target_mean, target_std, shift):
    """Calculate lognormal μ, σ from shifted lognormal parameters."""
    from scipy.optimize import fsolve
    
    def solve_lognorm_params(params):
        mu, sigma = params
        mean_ln = np.exp(mu + sigma**2/2)
        var_ln = np.exp(2*mu + sigma**2) * (np.exp(sigma**2) - 1)
        return [shift + mean_ln - target_mean, var_ln - target_std**2]
    
    mu_ln, sigma_ln = fsolve(solve_lognorm_params, [3.2, 0.5])
    return mu_ln, sigma_ln, shift


def get_truncnorm_params(mean, std, low, high):
    """Get standard normal bounds for truncated normal."""
    a = (low - mean) / std
    b = (high - mean) / std
    return a, b, mean, std


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
YAML_INPUT   = os.path.join(_HERE, "uqpce_input.yaml")
MATRIX_FILE  = os.path.join(_HERE, "uqpce_run_matrix.dat")


# ---------------------------------------------------------------------------
# UQPCE setup with TWO uncertain variables
# ---------------------------------------------------------------------------
def setup_and_init_uqpce() -> dict:
    """
    UQPCE uses 2D Normal distribution (optimal Hermite polynomials).
    Transform to:
      - t_hover: shifted lognormal
      - eta_hover: truncated normal (via inverse CDF transform)
    """
    
    # --- t_hover parameters (shifted lognormal) ---
    t_shift = 25.0
    t_target_mean = 55.0
    t_target_std = 18.0
    t_mu_ln, t_sigma_ln, t_shift = get_shifted_lognormal_params(
        t_target_mean, t_target_std, t_shift
    )
    
    # --- eta_hover parameters (truncated normal) ---
    eta_mean = 0.65
    eta_std = 0.05
    eta_lo = 0.55
    eta_hi = 0.75
    eta_a, eta_b, eta_loc, eta_scale = get_truncnorm_params(
        eta_mean, eta_std, eta_lo, eta_hi
    )
    
    print(f"  t_hover transform: shift={t_shift}, μ={t_mu_ln:.4f}, σ={t_sigma_ln:.4f}")
    print(f"    → mean={t_shift + np.exp(t_mu_ln + t_sigma_ln**2/2):.1f}s, std={t_target_std}s")
    print(f"  eta_hover transform: truncated N({eta_mean}, {eta_std}) on [{eta_lo}, {eta_hi}]")
    
    # UQPCE uses 2D STANDARD NORMAL distribution
    config = {
        "Variable 0": {
            "name": "z_t_hover",
            "distribution": "normal",
            "mean": 0.0,
            "stdev": 1.0,
            "type": "aleatory",
        },
        "Variable 1": {
            "name": "z_eta_hover",
            "distribution": "normal",
            "mean": 0.0,
            "stdev": 1.0,
            "type": "aleatory",
        },
        "Settings": {
            "order": 5,  # Reduced from 6 for 2D (curse of dimensionality)
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
    d["t_lognormal_params"] = {"mu": t_mu_ln, "sigma": t_sigma_ln, "shift": t_shift}
    d["eta_truncnorm_params"] = {"a": eta_a, "b": eta_b, "loc": eta_loc, "scale": eta_scale}
    
    return d


# ---------------------------------------------------------------------------
# Inner solver (with BOTH uncertainties)
# ---------------------------------------------------------------------------
def inner_solve_for_Wtotal(
    prob: om.Problem,
    t: float,
    eta: float,
    payload: float,
    range_m: float,
    n_c: int,
    dvars: tuple[float, float, float, float],
) -> dict:
    V, r, J, Sw = dvars
    
    # Store original values
    _orig_t = getattr(sc, "T_HOVER", None)
    _orig_eta = getattr(sc, "ETA_HOVER", None)
    
    sc.T_HOVER = float(t)
    sc.ETA_HOVER = float(eta)

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
                raise om.AnalysisError(f"No sign change for t={t:.1f}s, eta={eta:.3f}")

        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)

        return {
            "W_total":      root,
            "cruise_CL":    float(prob.get_val("cruise_CL")[0]),
            "disk_loading": float(prob.get_val("disk_loading")[0]),
            "blade_loading": float(prob.get_val("blade_loading")[0]),
        }

    except Exception as exc:
        print(f"  [inner_solve] FAILED t={t:.1f}s, eta={eta:.3f}: {exc}")
        return None

    finally:
        # Restore original values
        if _orig_t is None:
            if hasattr(sc, "T_HOVER"):
                del sc.T_HOVER
        else:
            sc.T_HOVER = _orig_t
        
        if _orig_eta is None:
            if hasattr(sc, "ETA_HOVER"):
                del sc.ETA_HOVER
        else:
            sc.ETA_HOVER = _orig_eta


# ---------------------------------------------------------------------------
# OpenMDAO component with 2D uncertainty
# ---------------------------------------------------------------------------
class QBiTUQComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("resp_cnt", types=int)
        self.options.declare("z_samples", types=np.ndarray)  # shape (n_quad, 2)
        self.options.declare("payload_kg", types=float)
        self.options.declare("range_m", types=float)
        self.options.declare("n_c", types=int)
        # t_hover params
        self.options.declare("t_mu_ln", types=float, default=3.28424)
        self.options.declare("t_sigma_ln", types=float, default=0.55403)
        self.options.declare("t_shift", types=float, default=25.0)
        # eta_hover params
        self.options.declare("eta_a", types=float, default=-2.0)
        self.options.declare("eta_b", types=float, default=2.0)
        self.options.declare("eta_loc", types=float, default=0.65)
        self.options.declare("eta_scale", types=float, default=0.05)

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
        
        z_samples = self.options["z_samples"]  # shape (n_quad, 2)
        
        # Transform standard normal samples to physical variables
        t_mu_ln = self.options["t_mu_ln"]
        t_sigma_ln = self.options["t_sigma_ln"]
        t_shift = self.options["t_shift"]
        
        eta_a = self.options["eta_a"]
        eta_b = self.options["eta_b"]
        eta_loc = self.options["eta_loc"]
        eta_scale = self.options["eta_scale"]
        
        # t_hover: shifted lognormal
        t_samples = t_shift + np.exp(t_mu_ln + t_sigma_ln * z_samples[:, 0])
        
        # eta_hover: truncated normal
        # Method 1: Using scipy.stats.norm and truncnorm (recommended)
        from scipy.stats import norm
        eta_dist = truncnorm(eta_a, eta_b, loc=eta_loc, scale=eta_scale)
        u_eta = norm.cdf(z_samples[:, 1])  # Standard normal → uniform
        eta_samples = eta_dist.ppf(u_eta)   # Uniform → truncated normal
        
        # Alternative Method 2: Direct transformation (if you prefer)
        # This is mathematically equivalent but avoids the intermediate uniform
        # from scipy.stats import norm
        # phi_z = norm.cdf(z_samples[:, 1])
        # eta_samples = eta_loc + eta_scale * norm.ppf(phi_z * (norm.cdf(eta_b) - norm.cdf(eta_a)) + norm.cdf(eta_a))
        # (Method 1 is cleaner)
        
        pl = self.options["payload_kg"]
        rm = self.options["range_m"]
        nc = self.options["n_c"]

        n_quad = len(t_samples)
        W_arr = np.empty(n_quad)
        cl_arr = np.empty(n_quad)
        dl_arr = np.empty(n_quad)
        bl_arr = np.empty(n_quad)

        W_PENALTY = float(W_TOTAL_BOUNDS[1])
        CL_PENALTY = CL_MAX * 1.5
        DL_PENALTY = DL_MAX * 1.5
        BL_PENALTY = BL_MAX * 1.5

        n_failed = 0
        for i in range(n_quad):
            res = inner_solve_for_Wtotal(
                self._inner, t_samples[i], eta_samples[i], pl, rm, nc, dvars
            )
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
            print(f"  [compute] {n_failed}/{n_quad} samples failed")

        outputs["W_total"] = W_arr
        outputs["cruise_CL"] = cl_arr
        outputs["disk_loading"] = dl_arr
        outputs["blade_loading"] = bl_arr


# ---------------------------------------------------------------------------
# Main optimisation
# ---------------------------------------------------------------------------
def run():
    total_start_time = time.time()
    
    print("=" * 70)
    print("UQPCE ROBUST DESIGN OPTIMIZATION WITH JOINT UNCERTAINTY")
    print("  • t_hover: shifted lognormal (aleatory)")
    print("  • eta_hover: truncated normal (epistemic)")
    print("=" * 70)
    
    # ---- 1. UQPCE initialisation ----------------------------------------
    print("\n[1/5] Setting up UQPCE...")
    setup_start = time.time()
    d = setup_and_init_uqpce()
    n_quad = d["resp_cnt"]
    z_samples = d["run_matrix"]  # shape (n_quad, 2)
    setup_time = time.time() - setup_start
    print(f"      ✓ Setup completed in {format_time(setup_time)}")
    print(f"      • PCE quadrature points: {n_quad}")
    print(f"      • 2D uncertainty space")
    
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
        t_mu_ln=d["t_lognormal_params"]["mu"],
        t_sigma_ln=d["t_lognormal_params"]["sigma"],
        t_shift=d["t_lognormal_params"]["shift"],
        eta_a=d["eta_truncnorm_params"]["a"],
        eta_b=d["eta_truncnorm_params"]["b"],
        eta_loc=d["eta_truncnorm_params"]["loc"],
        eta_scale=d["eta_truncnorm_params"]["scale"],
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
    
    prob.model.add_objective("W_total:mean")
    
    prob.setup()
    print("      ✓ Problem built successfully")
    
    # ---- 3. Run optimisation --------------------------------------------
    print("\n[3/5] Running robust optimization...")
    print("-" * 50)
    opt_start_time = time.time()
    prob.run_driver()
    opt_end_time = time.time()
    opt_duration = opt_end_time - opt_start_time
    print("-" * 50)
    
    # ---- 4. Results -----------------------------------------------------
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
    
    if hasattr(prob.driver, "iter_count"):
        print(f"\n  Optimization iterations: {prob.driver.iter_count}")
    
    total_duration = time.time() - total_start_time
    print(f"\n  Total runtime: {format_time(total_duration)}")
    print("=" * 60)


if __name__ == "__main__":
    run()