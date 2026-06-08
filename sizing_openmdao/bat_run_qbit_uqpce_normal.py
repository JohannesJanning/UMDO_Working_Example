"""
UQPCE-based robust MDO for the QBiT UAV sizing model.

Minimises the 95th-percentile upper CI of W_total (MTOM) subject to
robust constraints on cruise CL, disk loading, and blade loading,
under lognormal uncertainty in battery specific energy (ρ_bat).
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
# UQPCE setup - Battery Density Uncertainty
# ---------------------------------------------------------------------------
def setup_and_init_uqpce() -> dict:
    """
    UQPCE uses Normal distribution (optimal Hermite polynomials).
    Transform to lognormal inside QBiTUQComp for battery specific energy.
    
    Battery specific energy parameters (from paper):
    - Median: 235 Wh/kg
    - 5th percentile: 150 Wh/kg
    - 95th percentile: 370 Wh/kg
    - Lognormal(μ=5.46, σ=0.28) where μ, σ are parameters of ln(ρ_bat)
    """
    
    # Lognormal parameters in log-space (from paper)
    mu_ln = 5.46      # mean of ln(ρ_bat)
    sigma_ln = 0.28   # std dev of ln(ρ_bat)
    
    # Verify the distribution matches intended percentiles
    from scipy.stats import lognorm
    dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
    median = dist.median()
    p5 = dist.ppf(0.05)
    p95 = dist.ppf(0.95)
    
    print(f"  Battery specific energy distribution (Lognormal):")
    print(f"    μ_ln = {mu_ln:.4f}, σ_ln = {sigma_ln:.4f}")
    print(f"    Median = {median:.1f} Wh/kg")
    print(f"    5th percentile = {p5:.1f} Wh/kg")
    print(f"    95th percentile = {p95:.1f} Wh/kg")
    
    # UQPCE uses STANDARD NORMAL distribution (optimal Hermite)
    config = {
        "Variable 0": {
            "name": "z_battery",
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
    d["lognormal_params"] = {"mu": mu_ln, "sigma": sigma_ln}
    
    return d


# ---------------------------------------------------------------------------
# Inner solver (modified for battery density)
# ---------------------------------------------------------------------------
def inner_solve_for_Wtotal(
    prob: om.Problem,
    rho_bat: float,  # now battery specific energy instead of hover time
    payload: float,
    range_m: float,
    n_c: int,
    dvars: tuple[float, float, float, float],
) -> dict:
    V, r, J, Sw = dvars
    _orig_rho = getattr(sc, "BATTERY_DENSITY", None)
    sc.BATTERY_DENSITY = float(rho_bat)  # Wh/kg

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
                raise om.AnalysisError(f"No sign change in weight residual for ρ_bat={rho_bat:.1f} Wh/kg")

        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)

        return {
            "W_total":      root,
            "cruise_CL":    float(prob.get_val("cruise_CL")[0]),
            "disk_loading": float(prob.get_val("disk_loading")[0]),
            "blade_loading": float(prob.get_val("blade_loading")[0]),
        }

    except Exception as exc:
        print(f"  [inner_solve] FAILED ρ_bat={rho_bat:.1f} Wh/kg: {exc}")
        return None

    finally:
        if _orig_rho is None:
            if hasattr(sc, "BATTERY_DENSITY"):
                del sc.BATTERY_DENSITY
        else:
            sc.BATTERY_DENSITY = _orig_rho


# ---------------------------------------------------------------------------
# OpenMDAO component - Battery Density Uncertainty
# ---------------------------------------------------------------------------
class QBiTUQComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("resp_cnt", types=int)
        self.options.declare("z_samples", types=np.ndarray)
        self.options.declare("payload_kg", types=float)
        self.options.declare("range_m", types=float)
        self.options.declare("n_c", types=int)
        self.options.declare("mu_ln", types=float, default=5.46)   # lognormal μ for ρ_bat
        self.options.declare("sigma_ln", types=float, default=0.28) # lognormal σ for ρ_bat

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
        
        # Transform: ρ_bat = exp(μ + σ·Z)  (lognormal)
        rho_bat_samples = np.exp(mu_ln + sigma_ln * z_samples)
        
        pl = self.options["payload_kg"]
        rm = self.options["range_m"]
        nc = self.options["n_c"]

        W_arr = np.empty(len(rho_bat_samples))
        cl_arr = np.empty(len(rho_bat_samples))
        dl_arr = np.empty(len(rho_bat_samples))
        bl_arr = np.empty(len(rho_bat_samples))

        W_PENALTY = float(W_TOTAL_BOUNDS[1])
        CL_PENALTY = CL_MAX * 1.5
        DL_PENALTY = DL_MAX * 1.5
        BL_PENALTY = BL_MAX * 1.5

        n_failed = 0
        for i, rho in enumerate(rho_bat_samples):
            res = inner_solve_for_Wtotal(self._inner, rho, pl, rm, nc, dvars)
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
            print(f"  [compute] {n_failed}/{len(rho_bat_samples)} samples failed")

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
    print("Battery Specific Energy Uncertainty (Lognormal)")
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
    print(f"      • Transform: ρ_bat = exp({lognormal_params['mu']:.4f} + {lognormal_params['sigma']:.4f}·Z)")
    
    # Verify distribution statistics
    from scipy.stats import lognorm
    dist = lognorm(s=lognormal_params['sigma'], scale=np.exp(lognormal_params['mu']))
    print(f"      • Resulting distribution:")
    print(f"        - Median: {dist.median():.1f} Wh/kg")
    print(f"        - Mean: {dist.mean():.1f} Wh/kg")
    print(f"        - Std: {dist.std():.1f} Wh/kg")
    print(f"        - 5th pct: {dist.ppf(0.05):.1f} Wh/kg")
    print(f"        - 95th pct: {dist.ppf(0.95):.1f} Wh/kg")
    
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
    )
    prob.model.add_subsystem("eval", comp, promotes=["*"])
    
    uq = UQPCEGroup(
        significance= d["significance"],
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
        "cruise_CL:ci_upper", "cruise_CL:mean",
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
    
    #prob.model.add_constraint("cruise_CL:ci_upper", upper=CL_MAX)
    prob.model.add_constraint("cruise_CL:mean", upper=CL_MAX)
    prob.model.add_constraint("disk_loading:ci_upper", upper=DL_MAX)
    prob.model.add_constraint("blade_loading:ci_upper", upper=BL_MAX)
    
    # Create an objective that combines mean and variance
    prob.model.add_subsystem('objective_comp',
        om.ExecComp('obj = 0.5 *mean + 0.5 * var',
                    mean={'units': 'N'}, var={'units': 'N**2'}),
        promotes_inputs=[('mean', 'W_total:mean'), ('var', 'W_total:variance')],
        promotes_outputs=[('obj', 'objective')])

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
    print(f"    cruise_CL_mean    : {prob.get_val('cruise_CL:mean')[0]:.4f}  ≤ {CL_MAX}")
    print(f"    cruise_CL_upper    : {prob.get_val('cruise_CL:ci_upper')[0]:.4f}  ≤ {CL_MAX}")
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
    
    # Import the corrected RobustOptimizer from run_qbit_robust.py
    from bat_run_qbit_MCS import RobustOptimizer, sample_battery_density
    
    # Create Monte Carlo validation object with battery density parameters
    MEDIAN_RHO = 235.0  # Wh/kg
    SIGMA_LN = 0.28
    
    # Use the same battery density distribution parameters
    uq_ref = RobustOptimizer(
        payload_kg=3.0, 
        range_m=15000.0, 
        n_c=2, 
        n_mc=10000,  # High-fidelity MC validation
        median_rho=MEDIAN_RHO,
        sigma_ln=SIGMA_LN,
        seed=42
    )
    
    # Generate MC samples for battery density (10000 samples for validation)
    uq_ref.mc_samples = sample_battery_density(
        uq_ref.n_mc, 
        uq_ref.median_rho, 
        uq_ref.sigma_ln,
        seed=42, 
        method="lhs"
    )
    
    # Get the optimal design variables from optimization result
    x_opt = [
        prob.get_val("V_inf")[0], 
        prob.get_val("r")[0],
        prob.get_val("J")[0], 
        prob.get_val("S_w")[0]
    ]
    
    print(f"  Running Monte Carlo validation with {uq_ref.n_mc} samples at optimal design...")
    mc_stats = uq_ref._mc_stats(x_opt)
    
    val_duration = time.time() - val_start_time
    
    if mc_stats and len(mc_stats["results"]) > 0:
        # Filter valid results
        valid_results = [r for r in mc_stats["results"] if r is not None]
        mc_W = np.array([r["W_total"] for r in valid_results])
        mc_mean = np.mean(mc_W)
        mc_std = np.std(mc_W)
        mc_95th = np.percentile(mc_W, 95)
        mc_97_5th = np.percentile(mc_W, 97.5)
        
        # Get PCE predictions
        pce_mean = prob.get_val("W_total:mean")[0]
        pce_std = prob.get_val("W_total:variance")[0]**0.5
        pce_95th = prob.get_val("W_total:ci_upper")[0]
        
        # Calculate errors
        mean_error = abs(pce_mean - mc_mean)
        mean_error_pct = (mean_error / mc_mean) * 100 if mc_mean > 0 else float('nan')
        std_error = abs(pce_std - mc_std)
        std_error_pct = (std_error / mc_std) * 100 if mc_std > 0 else float('nan')
        p95_error = abs(pce_95th - mc_95th)
        p95_error_pct = (p95_error / mc_95th) * 100 if mc_95th > 0 else float('nan')
        
        print(f"\n  {'='*50}")
        print(f"  PCE vs MONTE CARLO VALIDATION (n={len(valid_results)} valid samples)")
        print(f"  {'='*50}")
        print(f"  Metric          | PCE            | MC             | Error")
        print(f"  {'-'*50}")
        print(f"  Mean (N)        | {pce_mean:10.2f}   | {mc_mean:10.2f}   | {mean_error:8.2f} N ({mean_error_pct:.1f}%)")
        print(f"  Std (N)         | {pce_std:10.2f}   | {mc_std:10.2f}   | {std_error:8.2f} N ({std_error_pct:.1f}%)")
        print(f"  95th pct (N)    | {pce_95th:10.2f}   | {mc_95th:10.2f}   | {p95_error:8.2f} N ({p95_error_pct:.1f}%)")
        print(f"  {'='*50}")
        print(f"  MTOM Mean (kg)  | {pce_mean/G:10.3f}   | {mc_mean/G:10.3f}   |")
        print(f"  MTOM 95th (kg)  | {pce_95th/G:10.3f}   | {mc_95th/G:10.3f}   |")
        print(f"  MTOM 97.5th (kg)| {pce_95th*1.02/G:10.3f}   | {mc_97_5th/G:10.3f}   |")
        
        # Check if PCE predictions are within acceptable tolerance
        if mean_error_pct < 5 and std_error_pct < 10 and p95_error_pct < 5:
            print(f"\n  ✓ PCE surrogate validation PASSED (errors within tolerance)")
        else:
            print(f"\n  ⚠ PCE surrogate validation WARNING (errors exceed tolerance)")
            if mean_error_pct >= 5:
                print(f"    - Mean error {mean_error_pct:.1f}% exceeds 5% tolerance")
            if std_error_pct >= 10:
                print(f"    - Std error {std_error_pct:.1f}% exceeds 10% tolerance")
            if p95_error_pct >= 5:
                print(f"    - 95th percentile error {p95_error_pct:.1f}% exceeds 5% tolerance")
    else:
        print(f"\n  ✗ Monte Carlo validation FAILED: No valid samples")
        print(f"    Check convergence of inner solver at optimal design")
    
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
    
    # ---- Uncertainty quantification summary ----------------------------
    print("\n" + "=" * 60)
    print("UNCERTAINTY CHARACTERIZATION")
    print("=" * 60)
    print("Battery specific energy is modeled as a lognormal uncertainty")
    print(f"with median {dist.median():.0f} Wh/kg and logarithmic standard")
    print(f"deviation {SIGMA_LN}, corresponding approximately to a 5–95%")
    print(f"interval of {dist.ppf(0.05):.0f}–{dist.ppf(0.95):.0f} Wh/kg.")
    print("This representation reflects the positive, right-skewed, and")
    print("multiplicative nature of effective pack-level energy density")
    print("at the conceptual design stage.")
    print("=" * 60)
    
    print("\n OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()