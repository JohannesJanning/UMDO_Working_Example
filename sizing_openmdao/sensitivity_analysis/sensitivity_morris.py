"""
sensitivity_morris.py - Morris sensitivity screening for QBiT.

This script screens uncertain QBiT parameters using Morris elementary effects.

Evaluator:
    W* = re-optimized deterministic MTOM

For each Morris sample, the script:
1. Sets the sampled qbit.constants values.
2. Reloads the QBiT model modules.
3. Runs the full OpenMDAO/SLSQP sizing optimization.
4. Returns optimized W_total.
5. Computes Morris mu* and sigma.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import qbit.constants as C
from qbit.constants import (
    G,
    W_TOTAL_BOUNDS,
    V_INF_BOUNDS,
    R_BOUNDS,
    J_BOUNDS,
    S_W_BOUNDS,
)


# ---------------------------------------------------------------------------
# Command-line settings
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Morris sensitivity screening for re-optimized QBiT MTOM."
)
parser.add_argument("--payload", type=float, default=3.0)
parser.add_argument("--range", type=float, default=15.0, help="One-way range [km]")
parser.add_argument("--nc", type=int, default=2)
parser.add_argument("--r", type=int, default=15, help="Number of Morris trajectories")
parser.add_argument("--p", type=int, default=6, help="Number of Morris grid levels")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-plot", action="store_true", help="Do not show plot")

args = parser.parse_args()

PAYLOAD_KG = args.payload
RANGE_M = args.range * 1_000.0
N_C = args.nc
R_MORRIS = args.r
P_MORRIS = args.p
SEED = args.seed


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------

# Relative perturbation for all parameters except T_HOVER
PARAM_PCT = 0.10  # ±10%


def rel_bounds(nominal: float, pct: float = PARAM_PCT, lb_hard=None, ub_hard=None):
    """Return relative bounds around nominal, with optional physical limits."""
    lb = nominal * (1.0 - pct)
    ub = nominal * (1.0 + pct)

    if lb_hard is not None:
        lb = max(lb, lb_hard)
    if ub_hard is not None:
        ub = min(ub, ub_hard)

    return lb, ub


PARAMS = [
    # attr              nominal    label                    type          lb/ub
    ("RHO_AIR",         1.225,    "rho_air",              "input",      *rel_bounds(1.225)),
    ("T_HOVER",         55.0,     "t_hover",              "input",      35.0, 75.0),
    ("BETA_QBIT",       0.18,     "beta_QBiT",            "param",      *rel_bounds(0.18, lb_hard=0.0, ub_hard=1.0)),
    ("ETA_HOVER",       0.65,     "eta_hover",            "param",      *rel_bounds(0.65, lb_hard=0.0, ub_hard=1.0)),
    ("CD0_WING",        0.01,     "CD0_wing",             "param",      *rel_bounds(0.01, lb_hard=0.0)),
    ("E_OSWALD",        0.80,     "e_Oswald",             "param",      *rel_bounds(0.80, lb_hard=0.0, ub_hard=1.0)),
    ("AR_FIXED",        8.0,      "AR",                   "input",      *rel_bounds(8.0, lb_hard=1.0)),
    ("SIGMA",           0.13,     "sigma_solidity",       "param",      *rel_bounds(0.13, lb_hard=0.0)),
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param",      *rel_bounds(0.012, lb_hard=0.0)),
    ("KAPPA_MAX",       1.15,     "kappa_max",            "param",      *rel_bounds(1.15, lb_hard=1.0)),
    ("BATTERY_DENSITY", 158.0,    "battery_density",      "input",      *rel_bounds(158.0, lb_hard=1.0)),
    ("BATTERY_EFF",     0.85,     "battery_eff",          "param",      *rel_bounds(0.85, lb_hard=0.0, ub_hard=1.0)),
    ("K_MOTOR",         2.506e-4, "k_motor",              "model",      *rel_bounds(2.506e-4, lb_hard=0.0)),
    ("K_ESC",           3.594e-4, "k_ESC",                "model",      *rel_bounds(3.594e-4, lb_hard=0.0)),
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model",      *rel_bounds(0.7484)),
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model",      *rel_bounds(0.0403, lb_hard=0.0)),
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model",      *rel_bounds(-0.0802)),
    ("K_WING_B",        2.2854,   "k_wing_B",             "model",      *rel_bounds(2.2854, lb_hard=0.0)),
    ("DL_MAX",          55.0,     "DL_MAX",               "constraint", *rel_bounds(55.0, lb_hard=0.0)),
    ("BL_MAX",          80.0,     "BL_MAX",               "constraint", *rel_bounds(80.0, lb_hard=0.0)),
    ("CL_MAX",          1.0,      "CL_MAX",               "constraint", *rel_bounds(1.0, lb_hard=0.0, ub_hard=2.0)),
]

K = len(PARAMS)
LB = np.array([p[4] for p in PARAMS], dtype=float)
UB = np.array([p[5] for p in PARAMS], dtype=float)


# ---------------------------------------------------------------------------
# Constant patching helpers
# ---------------------------------------------------------------------------

def restore_constants() -> None:
    """Reset all screened constants to their nominal values."""
    for attr, nominal, *_ in PARAMS:
        setattr(C, attr, nominal)


def set_constants_from_array(x: np.ndarray) -> None:
    """Set qbit.constants values from one physical parameter vector."""
    for j, (attr, *_rest) in enumerate(PARAMS):
        setattr(C, attr, float(x[j]))


def reload_qbit_modules() -> None:
    """
    Reload qbit modules so patched constants are picked up.

    This is needed because some modules may import constants directly.
    """
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# OpenMDAO deterministic optimization evaluator
# ---------------------------------------------------------------------------

def optimise_W(payload_kg: float, range_m: float, n_c: int) -> float:
    """
    Run the deterministic OpenMDAO sizing optimization.

    Returns:
        Optimized total weight W_total [N].
    """

    reload_qbit_modules()

    from qbit.models.qbit_model import build_qbit_model

    prob = om.Problem(reports=None)
    prob.model = build_qbit_model(payload_kg, range_m, n_c)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["maxiter"] = 2000
    prob.driver.options["disp"] = False

    prob.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var("V_inf", lower=V_INF_BOUNDS[0], upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r", lower=R_BOUNDS[0], upper=R_BOUNDS[1])
    prob.model.add_design_var("J", lower=J_BOUNDS[0], upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w", lower=S_W_BOUNDS[0], upper=S_W_BOUNDS[1])

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading", upper=C.DL_MAX)
    prob.model.add_constraint("blade_loading", upper=C.BL_MAX)
    prob.model.add_constraint("cruise_CL", upper=C.CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")

    prob.setup()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()

    return float(prob.get_val("W_total")[0])


def evaluate_parameter_sample(x: np.ndarray) -> float:
    """
    Evaluate one Morris sample.

    The sampled constants are patched into qbit.constants, the deterministic
    sizing optimization is rerun, and the optimized W_total is returned.
    """

    restore_constants()
    set_constants_from_array(x)

    try:
        W = optimise_W(PAYLOAD_KG, RANGE_M, N_C)

        if W < 1.0 * G or W > 500.0 * G:
            return float("nan")

        return W

    except Exception:
        return float("nan")

    finally:
        restore_constants()


def compute_baseline() -> float:
    """Compute nominal optimized MTOM."""
    restore_constants()
    return optimise_W(PAYLOAD_KG, RANGE_M, N_C)


# ---------------------------------------------------------------------------
# Morris sampling
# ---------------------------------------------------------------------------

def morris_trajectory(k: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Morris trajectory in [0, 1]^k.

    Successive points differ in exactly one coordinate.
    """

    delta = p / (2.0 * (p - 1))

    D_star = np.diag(2 * rng.integers(0, 2, size=k) - 1)
    perm = rng.permutation(k)
    P_star = np.eye(k)[perm]

    x_star = rng.uniform(0.0, 1.0 - delta, size=k)

    B = np.tril(np.ones((k + 1, k)), -1)
    J_col = np.ones((k + 1, 1))
    J_row = np.ones((1, k))

    B_star = (
        J_col @ x_star.reshape(1, k)
        + (delta / 2.0) * ((2.0 * B - J_col @ J_row) @ D_star @ P_star.T)
    )

    return np.clip(B_star, 0.0, 1.0)


def normalised_to_physical(X_norm: np.ndarray) -> np.ndarray:
    """Map [0,1] Morris samples to physical parameter bounds."""
    return LB + X_norm * (UB - LB)


def run_morris(W_base: float) -> list[dict]:
    """Run Morris elementary-effects screening."""

    rng = np.random.default_rng(SEED)
    n_total = R_MORRIS * (K + 1)

    print("\n" + "=" * 70)
    print("MORRIS ELEMENTARY-EFFECTS SCREENING")
    print("=" * 70)
    print(f"Parameters          : {K}")
    print(f"Trajectories        : {R_MORRIS}")
    print(f"Grid levels         : {P_MORRIS}")
    print(f"Optimizer calls     : {n_total}")
    print(f"Baseline W*         : {W_base / G:.4f} kg")
    print("=" * 70)

    ee_raw = [[] for _ in range(K)]
    n_done = 0
    n_fail = 0
    start = time.time()

    for tr in range(R_MORRIS):
        X_norm = morris_trajectory(K, P_MORRIS, rng)
        X_phys = normalised_to_physical(X_norm)

        Y = np.full(K + 1, np.nan)

        for i in range(K + 1):
            n_done += 1
            elapsed = time.time() - start
            eta = elapsed / n_done * (n_total - n_done) if n_done > 0 else 0.0

            print(
                f"\r  [{n_done:>4}/{n_total}] "
                f"trajectory {tr + 1:>2}/{R_MORRIS}, "
                f"point {i + 1:>2}/{K + 1}, "
                f"ETA {eta / 60:.1f} min",
                end="",
                flush=True,
            )

            Y[i] = evaluate_parameter_sample(X_phys[i])

            if np.isnan(Y[i]):
                n_fail += 1

        for step in range(K):
            diff = X_norm[step + 1] - X_norm[step]
            j = int(np.argmax(np.abs(diff)))

            delta_phys = X_phys[step + 1, j] - X_phys[step, j]

            if abs(delta_phys) < 1e-15:
                continue

            if np.isnan(Y[step]) or np.isnan(Y[step + 1]):
                continue

            ee = (Y[step + 1] - Y[step]) / delta_phys
            ee_raw[j].append(ee)

    print()
    print(f"Morris evaluations complete: {n_done} runs, {n_fail} failures")

    results = []

    for j, (attr, nominal, label, typ, lb, ub) in enumerate(PARAMS):
        vals = np.array(ee_raw[j], dtype=float)

        if vals.size == 0:
            mu_star = 0.0
            sigma = 0.0
            n_ee = 0
        else:
            parameter_range = ub - lb
            ee_norm = vals * parameter_range / W_base

            mu_star = float(np.mean(np.abs(ee_norm)))
            sigma = float(np.std(ee_norm, ddof=0))
            n_ee = int(vals.size)

        results.append(
            {
                "attr": attr,
                "label": label,
                "type": typ,
                "nominal": nominal,
                "mu_star": mu_star,
                "sigma": sigma,
                "n_ee": n_ee,
            }
        )

    results.sort(key=lambda r: r["mu_star"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_morris_table(results: list[dict], W_base: float) -> None:
    """Print Morris results table without recommendation labels."""

    print("\n" + "=" * 82)
    print("MORRIS SCREENING RESULTS")
    print("=" * 82)
    print(f"Response    : re-optimized W_total")
    print(f"Baseline W*: {W_base / G:.4f} kg")
    print(f"Mission    : payload={PAYLOAD_KG:g} kg, range={RANGE_M / 1e3:.0f} km, n_c={N_C}")
    print("=" * 82)

    print(
        f"{'Rank':>4}  {'Parameter':28s}  {'Type':8s}  "
        f"{'mu*':>10}  {'sigma':>10}  {'sigma/mu*':>10}  {'n_EE':>5}"
    )
    print("-" * 82)

    for i, r in enumerate(results, 1):
        ratio = r["sigma"] / r["mu_star"] if r["mu_star"] > 1e-12 else 0.0

        print(
            f"{i:>4}  {r['label']:28s}  {r['type']:8s}  "
            f"{r['mu_star']:>10.4f}  {r['sigma']:>10.4f}  "
            f"{ratio:>10.2f}  {r['n_ee']:>5}"
        )

    print("=" * 82)
    print("mu*       : mean absolute elementary effect, normalized by parameter range and W*")
    print("sigma     : spread of elementary effects; high values indicate nonlinearity/interactions")
    print("sigma/mu*: rough nonlinearity/interactions indicator")
    print("=" * 82)


def plot_morris(results: list[dict], W_base: float) -> None:
    """Show Morris ranking plot directly."""

    labels = [r["label"] for r in results]
    mu_star = np.array([r["mu_star"] for r in results])
    sigma = np.array([r["sigma"] for r in results])

    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.42 * len(labels))))

    ax = axes[0]
    ax.barh(y, mu_star)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mu*")
    ax.set_title("Morris importance ranking")

    ax = axes[1]
    ax.scatter(mu_star, sigma)
    for label, x, yv in zip(labels, mu_star, sigma):
        if x > 0.02 or yv > 0.02:
            ax.annotate(label, (x, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)

    lim = max(float(np.max(mu_star)), float(np.max(sigma)), 1e-3) * 1.15
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mu*")
    ax.set_ylabel("sigma")
    ax.set_title("Importance vs nonlinearity/interactions")

    fig.suptitle(
        f"Morris screening, W*_base = {W_base / G:.3f} kg",
        fontsize=13,
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)

    total_start = time.time()

    print("\nQBiT MORRIS SENSITIVITY SCREENING")
    print("=" * 70)
    print(f"Mission    : payload={PAYLOAD_KG:g} kg, range={RANGE_M / 1e3:.0f} km, n_c={N_C}")
    print(f"Evaluator  : re-optimized W_total using OpenMDAO/SLSQP")
    print(f"Parameters : {K}")
    print(f"Seed       : {SEED}")
    print("=" * 70)

    print("\nComputing nominal baseline...")
    baseline_start = time.time()
    W_base = compute_baseline()
    baseline_time = time.time() - baseline_start

    print(f"Baseline W*: {W_base / G:.4f} kg ({baseline_time:.1f} s)")

    if W_base < 1.0 * G or W_base > 500.0 * G:
        raise RuntimeError(f"Baseline W* is physically unreasonable: {W_base / G:.2f} kg")

    results = run_morris(W_base)

    print_morris_table(results, W_base)

    if not args.no_plot:
        plot_morris(results, W_base)

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Baseline time : {baseline_time:.1f} s")
    print(f"Total runtime : {total_time:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()