"""
run_qbit_uqpce.py - UQPCE-based robust MDO for the QBiT UAV sizing model.

The script optimizes the QBiT design under uncertainty in hover time.

Workflow:
1. Generate UQPCE samples for a standard normal variable Z.
2. Transform Z into lognormal hover-time samples.
3. Evaluate the fixed QBiT design for each hover-time sample.
4. Use UQPCE to estimate mean, variance, and upper confidence value.
5. Optimize the design variables against the robust MTOM objective.
6. Validate the final design with Monte Carlo.
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import openmdao.api as om
import yaml
from scipy.optimize import fsolve

from run_qbit import evaluate_qbit_at_design
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (
    G,
    W_TOTAL_BOUNDS,
    V_INF_BOUNDS,
    R_BOUNDS,
    J_BOUNDS,
    S_W_BOUNDS,
    DL_MAX,
    BL_MAX,
    CL_MAX,
)

from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.pce.pce import PCE
from uqpce.pce.io import read_input_file
from uqpce.mdao import interface


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Mission definition
PAYLOAD_KG = 3.0
RANGE_M = 15_000.0          # one-way depot-to-customer distance [m]
N_CUSTOMERS = 2

# Hover-time requirement uncertainty
T_HOVER_SHIFT = 25.0        # lower bound / shift [s]
T_HOVER_MEAN = 55.0         # target mean hover time [s]
T_HOVER_STD = 18.0          # target standard deviation [s]

# Optimizer settings
OPT_MAXITER = 200
OPT_TOL = 1e-6

# Seed and Monte Carlo validation settings
MC_VALIDATION_SAMPLES = 2000
PCE_SEED = 42
MC_SEED = 123



# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
YAML_INPUT = os.path.join(_HERE, "uqpce_input.yaml")
MATRIX_FILE = os.path.join(_HERE, "uqpce_run_matrix.dat")


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Return runtime in a readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes ({seconds:.0f} seconds)"

    hours = seconds / 3600
    minutes = (seconds % 3600) / 60
    return f"{hours:.1f} hours ({int(hours)}h {int(minutes)}m)"


def shifted_lognormal_params(
    target_mean: float,
    target_std: float,
    shift: float,
) -> tuple[float, float]:
    """
    Compute mu and sigma for:

        t_hover = shift + exp(mu + sigma * Z)

    where Z is standard normal.
    """

    def residual(params):
        mu, sigma = params

        mean_ln = np.exp(mu + sigma**2 / 2)
        var_ln = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)

        return [
            shift + mean_ln - target_mean,
            var_ln - target_std**2,
        ]

    mu_ln, sigma_ln = fsolve(residual, [3.2, 0.5])
    return float(mu_ln), float(sigma_ln)


# ---------------------------------------------------------------------------
# UQPCE setup
# ---------------------------------------------------------------------------

def setup_and_init_uqpce(seed: int = PCE_SEED) -> dict:
    """
    Create the UQPCE input files and initialize the PCE dictionary.

    UQPCE samples a standard normal variable Z.
    The QBiT component later transforms Z into hover time.
    """

    np.random.seed(seed)

    shift = T_HOVER_SHIFT
    target_mean = T_HOVER_MEAN
    target_std = T_HOVER_STD

    mu_ln, sigma_ln = shifted_lognormal_params(
        target_mean=target_mean,
        target_std=target_std,
        shift=shift,
    )

    print("  Hover-time uncertainty model:")
    print(f"    t_hover = {shift:.1f} + exp({mu_ln:.4f} + {sigma_ln:.4f} * Z)")
    print(f"    mean = {target_mean:.1f} s, std = {target_std:.1f} s")

    # UQPCE input file: one aleatory standard-normal variable.
    config = {
        "Variable 0": {
            "name": "z_hover",
            "distribution": "normal",
            "mean": 0.0,
            "stdev": 1.0,
            "type": "aleatory",
        },
        "Settings": {
            "order": 7,
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

    # Generate the UQPCE sample matrix.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        pce = PCE(outputs=False, plot=False, verbose=False, **settings)

        for value in var_dict.values():
            pce.add_variable(**value)

        X = pce.sample()
        np.savetxt(MATRIX_FILE, X)

    # UQPCE reads its own generated input files through this interface.
    d = interface.initialize_dict(YAML_INPUT, MATRIX_FILE)

    # Store the lognormal transform parameters for later use.
    d["lognormal_params"] = {
        "mu": mu_ln,
        "sigma": sigma_ln,
        "shift": shift,
    }

    return d


# ---------------------------------------------------------------------------
# QBiT uncertainty component
# ---------------------------------------------------------------------------

class QBiTUQComp(om.ExplicitComponent):
    """
    OpenMDAO component that evaluates one design over all UQPCE samples.

    Inputs:
        V_inf, r, J, S_w

    Outputs:
        sample arrays for W_total, cruise_CL, disk_loading, blade_loading
    """

    def initialize(self):
        self.options.declare("resp_cnt", types=int)
        self.options.declare("z_samples", types=np.ndarray)

        self.options.declare("payload_kg", types=float)
        self.options.declare("range_m", types=float)
        self.options.declare("n_c", types=int)

        self.options.declare("mu_ln", types=float)
        self.options.declare("sigma_ln", types=float)
        self.options.declare("shift", types=float)

    def setup(self):
        # Design variables passed in from the outer optimizer.
        self.add_input("V_inf", val=35.0, units="m/s")
        self.add_input("r", val=0.20, units="m")
        self.add_input("J", val=1.3)
        self.add_input("S_w", val=0.30, units="m**2")

        n = self.options["resp_cnt"]

        # One output value per UQPCE sample.
        self.add_output("W_total", shape=(n,), units="N")
        self.add_output("cruise_CL", shape=(n,))
        self.add_output("disk_loading", shape=(n,), units="N/m**2")
        self.add_output("blade_loading", shape=(n,))

        # Finite differences are used by the outer SLSQP optimizer.
        self.declare_partials("*", "*", method="fd", step=1e-4, step_calc="rel")

        # Build one reusable inner QBiT model.
        # Only W_total is solved inside evaluate_qbit_at_design().
        self._inner = om.Problem(reports=None)
        self._inner.model = build_qbit_model(
            self.options["payload_kg"],
            self.options["range_m"],
            self.options["n_c"],
        )

        self._inner.model.set_input_defaults("W_total", val=6.0 * G, units="N")
        self._inner.model.set_input_defaults("V_inf", val=35.0, units="m/s")
        self._inner.model.set_input_defaults("r", val=0.20, units="m")
        self._inner.model.set_input_defaults("J", val=1.3)
        self._inner.model.set_input_defaults("S_w", val=0.30, units="m**2")

        self._inner.setup()
        self._inner.run_model()

    def compute(self, inputs, outputs):
        # Current design proposed by the optimizer.
        dvars = (
            float(inputs["V_inf"][0]),
            float(inputs["r"][0]),
            float(inputs["J"][0]),
            float(inputs["S_w"][0]),
        )

        # Transform standard-normal UQPCE samples into hover-time samples.
        z_samples = self.options["z_samples"]
        mu_ln = self.options["mu_ln"]
        sigma_ln = self.options["sigma_ln"]
        shift = self.options["shift"]

        t_samples = shift + np.exp(mu_ln + sigma_ln * z_samples)

        W_arr = np.empty(len(t_samples))
        cl_arr = np.empty(len(t_samples))
        dl_arr = np.empty(len(t_samples))
        bl_arr = np.empty(len(t_samples))

        # Penalties keep the optimizer away from designs that fail analysis.
        W_PENALTY = float(W_TOTAL_BOUNDS[1])
        CL_PENALTY = CL_MAX * 1.5
        DL_PENALTY = DL_MAX * 1.5
        BL_PENALTY = BL_MAX * 1.5

        n_failed = 0

        for i, t_hover in enumerate(t_samples):
            res = evaluate_qbit_at_design(
                prob=self._inner,
                t_hover=float(t_hover),
                dvars=dvars,
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
            print(f"  [QBiTUQComp] {n_failed}/{len(t_samples)} samples failed")

        outputs["W_total"] = W_arr
        outputs["cruise_CL"] = cl_arr
        outputs["disk_loading"] = dl_arr
        outputs["blade_loading"] = bl_arr


# ---------------------------------------------------------------------------
# OpenMDAO robust optimization problem
# ---------------------------------------------------------------------------

def build_robust_problem(d: dict) -> om.Problem:
    """
    Build the outer robust optimization problem.

    The optimizer changes V_inf, r, J, and S_w.
    UQPCE computes statistics of the sample responses.
    """

    z_samples = d["run_matrix"][:, 0]
    lognormal_params = d["lognormal_params"]
    n_quad = d["resp_cnt"]

    prob = om.Problem(reports=None)

    # Independent design variables.
    ivc = om.IndepVarComp()
    ivc.add_output("V_inf", val=35.0, units="m/s")
    ivc.add_output("r", val=0.20, units="m")
    ivc.add_output("J", val=1.3)
    ivc.add_output("S_w", val=0.30, units="m**2")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    # Evaluates QBiT at all UQPCE samples.
    qbit_eval = QBiTUQComp(
        resp_cnt=n_quad,
        z_samples=z_samples,
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS,
        mu_ln=lognormal_params["mu"],
        sigma_ln=lognormal_params["sigma"],
        shift=lognormal_params["shift"],
    )
    prob.model.add_subsystem("eval", qbit_eval, promotes=["*"])

    # Converts sample arrays into PCE statistics.
    uq = UQPCEGroup(
        significance=d["significance"],
        var_basis=d["var_basis"],
        norm_sq=d["norm_sq"],
        resampled_var_basis=d["resampled_var_basis"],
        tail="upper",
        epistemic_cnt=d["epistemic_cnt"],
        aleatory_cnt=d["aleatory_cnt"],
        uncert_list=["W_total", "cruise_CL", "disk_loading", "blade_loading"],
        tanh_omega=0.01,
        sample_ref0=[63.16, 0.494, 76.5, 0.0130],
        sample_ref=[2.53, 0.018, 5.5, 0.0002],
    )

    promoted = [
        "W_total",
        "cruise_CL",
        "disk_loading",
        "blade_loading",
        "W_total:ci_upper",
        "W_total:mean",
        "W_total:variance",
        "cruise_CL:ci_upper",
        "cruise_CL:mean",
        "disk_loading:ci_upper",
        "disk_loading:mean",
        "blade_loading:ci_upper",
        "blade_loading:mean",
    ]

    prob.model.add_subsystem("uq", uq, promotes=promoted)

    # SLSQP optimizes the robust objective using finite differences.
    prob.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP", 
        maxiter=OPT_MAXITER, 
        tol=OPT_TOL)
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = []

    # Outer design variables.
    prob.model.add_design_var("V_inf", lower=V_INF_BOUNDS[0], upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r", lower=R_BOUNDS[0], upper=R_BOUNDS[1])
    prob.model.add_design_var("J", lower=J_BOUNDS[0], upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w", lower=S_W_BOUNDS[0], upper=S_W_BOUNDS[1])

    # Mean constraints are used here.
    # Replace with :ci_upper if you want robust constraints.
    prob.model.add_constraint("cruise_CL:mean", upper=CL_MAX)
    prob.model.add_constraint("disk_loading:mean", upper=DL_MAX)
    prob.model.add_constraint("blade_loading:mean", upper=BL_MAX)

    # Robust objective: minimize upper confidence estimate of MTOM.
    prob.model.add_objective("W_total:ci_upper")

    prob.setup()

    return prob


# ---------------------------------------------------------------------------
# Reporting and validation
# ---------------------------------------------------------------------------

def print_initial_pce_check(prob: om.Problem) -> None:
    """Run the model once and print initial PCE statistics."""
    prob.run_model()

    pce_mean = prob.get_val("W_total:mean")[0]
    pce_var = prob.get_val("W_total:variance")[0]
    pce_ci = prob.get_val("W_total:ci_upper")[0]

    print("\nInitial design PCE check:")
    print(f"  Mean MTOM   : {pce_mean / G:.3f} kg")
    print(f"  Std MTOM    : {pce_var**0.5:.2f} N")
    print(f"  Robust MTOM : {pce_ci / G:.3f} kg")


def print_results(prob: om.Problem) -> None:
    """Print final robust optimization results."""
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
    print(f"  Mean MTOM      : {prob.get_val('W_total:mean')[0] / G:.3f} kg")
    print(f"  MTOM variance  : {prob.get_val('W_total:variance')[0]:.4f} N²")
    print(f"  Robust MTOM    : {prob.get_val('W_total:ci_upper')[0] / G:.3f} kg")

    print()
    print("  Constraint values:")
    print(f"    cruise_CL_mean    : {prob.get_val('cruise_CL:mean')[0]:.4f} ≤ {CL_MAX}")
    print(f"    disk_loading_mean : {prob.get_val('disk_loading:mean')[0]:.2f} N/m² ≤ {DL_MAX}")
    print(f"    blade_loading_mean: {prob.get_val('blade_loading:mean')[0]:.4f} ≤ {BL_MAX}")

    if hasattr(prob.driver, "iter_count"):
        print(f"\n  Optimization iterations: {prob.driver.iter_count}")

    if hasattr(prob.driver, "result") and hasattr(prob.driver.result, "nfev"):
        print(f"  Function evaluations: {prob.driver.result.nfev}")

    x_opt = [
        float(prob.get_val("V_inf")[0]),
        float(prob.get_val("r")[0]),
        float(prob.get_val("J")[0]),
        float(prob.get_val("S_w")[0]),
    ]

    print()
    print(f"  Design vector  : {x_opt}")

    print("=" * 60)


def validate_with_monte_carlo(prob: om.Problem) -> float:
    """
    Validate the final PCE result with Monte Carlo samples.

    This uses the existing Monte Carlo implementation from run_qbit_monte_carlo.py.
    """
    start = time.time()

    from run_qbit_monte_carlo import RobustOptimizer, sample_t_hover

    uq_ref = RobustOptimizer(
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS,
        n_mc=MC_VALIDATION_SAMPLES,
        seed=MC_SEED,
    )

    uq_ref.mc_samples = sample_t_hover(
        MC_VALIDATION_SAMPLES,
        uq_ref.mean_t,
        uq_ref.std_t,
        uq_ref.shift_t,
        seed=MC_SEED,
    )

    x_opt = [
        prob.get_val("V_inf")[0],
        prob.get_val("r")[0],
        prob.get_val("J")[0],
        prob.get_val("S_w")[0],
    ]

    mc_stats = uq_ref._mc_stats(x_opt)

    if mc_stats is None:
        print("\nMonte Carlo validation failed: at least one MC sample did not solve.")
        return time.time() - start

    if mc_stats:
        mc_W = [res["W_total"] for res in mc_stats["results"] if res]

        mc_mean = np.mean(mc_W)
        mc_std = np.std(mc_W)

        pce_mean = prob.get_val("W_total:mean")[0]
        pce_std = prob.get_val("W_total:variance")[0] ** 0.5

        print("\nMonte Carlo validation:")
        print(
            f"  PCE mean = {pce_mean:.2f} N, "
            f"MC mean = {mc_mean:.2f} N, "
            f"error = {abs(pce_mean - mc_mean) / mc_mean * 100:.1f}%"
        )
        print(
            f"  PCE std  = {pce_std:.2f} N, "
            f"MC std  = {mc_std:.2f} N, "
            f"error = {abs(pce_std - mc_std) / mc_std * 100:.1f}%"
        )

    return time.time() - start


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def run() -> None:
    """Run the full UQPCE robust optimization workflow."""
    total_start = time.time()

    print("=" * 70)
    print("UQPCE ROBUST DESIGN OPTIMIZATION")
    print("=" * 70)

    print("\n[1/5] Setting up UQPCE...")
    setup_start = time.time()
    d = setup_and_init_uqpce(seed=PCE_SEED)
    setup_time = time.time() - setup_start
    print(f"      Setup completed in {format_time(setup_time)}")
    print(f"      PCE sample points: {d['resp_cnt']}")

    print("\n[2/5] Building robust OpenMDAO problem...")
    prob = build_robust_problem(d)
    print("      Problem built successfully")

    print("\n[3/5] Checking PCE at initial design...")
    print_initial_pce_check(prob)

    print("\n[4/5] Running robust optimization...")
    opt_start = time.time()
    prob.run_driver()
    opt_time = time.time() - opt_start

    print_results(prob)

    print("\n[5/5] Validating final design with Monte Carlo...")
    val_time = validate_with_monte_carlo(prob)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"PCE setup time    : {format_time(setup_time)}")
    print(f"Optimization time : {format_time(opt_time)}")
    print(f"Validation time   : {format_time(val_time)}")
    print(f"Total runtime     : {format_time(total_time)}")
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    run()