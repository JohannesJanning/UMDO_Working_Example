"""
sensitivity_sobol.py - Sobol sensitivity screening for QBiT.

This script screens uncertain QBiT parameters using Sobol variance-based indices.

Evaluator:
    W* = re-optimized deterministic MTOM

For each Sobol sample, the script:
1. Sets the sampled qbit.constants values.
2. Reloads the QBiT model modules.
3. Runs the full OpenMDAO/SLSQP sizing optimization.
4. Returns optimized W_total.
5. Computes Sobol S1 and ST indices.

Requires:
    pip install SALib
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
    description="Sobol sensitivity screening for re-optimized QBiT MTOM."
)
parser.add_argument("--payload", type=float, default=3.0)
parser.add_argument("--range", type=float, default=15.0, help="One-way range [km]")
parser.add_argument("--nc", type=int, default=2)
parser.add_argument("--n", type=int, default=64, help="Sobol base sample count")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-plot", action="store_true", help="Do not show plot")

args = parser.parse_args()

PAYLOAD_KG = args.payload
RANGE_M = args.range * 1_000.0
N_C = args.nc
N_SOBOL = args.n
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
    """Reload qbit modules so patched constants are picked up."""
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
    """Run deterministic OpenMDAO sizing optimization and return W_total [N]."""

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
    """Patch constants, re-optimize QBiT, and return optimized W_total [N]."""

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
# Sobol analysis
# ---------------------------------------------------------------------------

def run_sobol(W_base: float) -> list[dict]:
    """Run Sobol variance-based sensitivity analysis."""

    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
    except ImportError as exc:
        raise ImportError("SALib is required. Install with: pip install SALib") from exc

    problem_unit = {
        "num_vars": K,
        "names": [p[0] for p in PARAMS],
        "bounds": [[0.0, 1.0]] * K,
    }

    print("\n" + "=" * 70)
    print("SOBOL VARIANCE-BASED SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Parameters          : {K}")
    print(f"Base sample N       : {N_SOBOL}")
    print(f"Second order        : False")
    print(f"Estimated calls     : {N_SOBOL * (2 * K + 2)}")
    print(f"Baseline W*         : {W_base / G:.4f} kg")
    print("=" * 70)

    try:
        X_unit = sobol_sample.sample(
            problem_unit,
            N_SOBOL,
            calc_second_order=False,
            seed=SEED,
        )
    except TypeError:
        np.random.seed(SEED)
        X_unit = sobol_sample.sample(
            problem_unit,
            N_SOBOL,
            calc_second_order=False,
        )

    X_phys = LB + X_unit * (UB - LB)

    Y = np.full(len(X_phys), np.nan)
    n_fail = 0
    start = time.time()

    for i, x in enumerate(X_phys):
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(X_phys) - i - 1) if i > 0 else 0.0

        print(
            f"\r  [{i + 1:>5}/{len(X_phys)}] "
            f"ETA {eta / 60:.1f} min",
            end="",
            flush=True,
        )

        Y[i] = evaluate_parameter_sample(x)

        if np.isnan(Y[i]):
            n_fail += 1

    print()
    print(f"Sobol evaluations complete: {len(Y)} runs, {n_fail} failures")

    if n_fail > 0:
        mean_Y = float(np.nanmean(Y))
        Y = np.where(np.isnan(Y), mean_Y, Y)
        print(f"NaN values replaced with mean response: {mean_Y / G:.4f} kg")

    Si = sobol_analyze.analyze(
        problem_unit,
        Y,
        calc_second_order=False,
        print_to_console=False,
        seed=SEED,
    )

    results = []

    for j, (attr, nominal, label, typ, lb, ub) in enumerate(PARAMS):
        results.append(
            {
                "attr": attr,
                "label": label,
                "type": typ,
                "nominal": nominal,
                "S1": float(Si["S1"][j]),
                "S1_conf": float(Si["S1_conf"][j]),
                "ST": float(Si["ST"][j]),
                "ST_conf": float(Si["ST_conf"][j]),
            }
        )

    results.sort(key=lambda r: r["ST"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sobol_table(results: list[dict], W_base: float) -> None:
    """Print Sobol result table without recommendation labels."""

    print("\n" + "=" * 96)
    print("SOBOL SENSITIVITY RESULTS")
    print("=" * 96)
    print("Response    : re-optimized W_total")
    print(f"Baseline W*: {W_base / G:.4f} kg")
    print(f"Mission    : payload={PAYLOAD_KG:g} kg, range={RANGE_M / 1e3:.0f} km, n_c={N_C}")
    print("=" * 96)

    print(
        f"{'Rank':>4}  {'Parameter':28s}  {'Type':11s}  "
        f"{'S1':>10}  {'S1_conf':>10}  {'ST':>10}  {'ST_conf':>10}  {'ST-S1':>10}"
    )
    print("-" * 96)

    for i, r in enumerate(results, 1):
        interaction = r["ST"] - r["S1"]

        print(
            f"{i:>4}  {r['label']:28s}  {r['type']:11s}  "
            f"{r['S1']:>10.4f}  {r['S1_conf']:>10.4f}  "
            f"{r['ST']:>10.4f}  {r['ST_conf']:>10.4f}  "
            f"{interaction:>10.4f}"
        )

    print("=" * 96)
    print("S1    : first-order Sobol index; main effect only")
    print("ST    : total-order Sobol index; main effect plus interactions")
    print("ST-S1 : interaction contribution estimate")
    print("=" * 96)


def plot_sobol(results: list[dict], W_base: float) -> None:
    """Show Sobol S1/ST bar plot directly."""

    labels = [r["label"] for r in results]
    S1 = np.array([r["S1"] for r in results])
    ST = np.array([r["ST"] for r in results])
    S1_conf = np.array([r["S1_conf"] for r in results])
    ST_conf = np.array([r["ST_conf"] for r in results])

    y = np.arange(len(labels))
    h = 0.36

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(labels))))

    ax.barh(y + h / 2, ST, height=h, label="ST")
    ax.barh(y - h / 2, S1, height=h, label="S1", alpha=0.55)

    ax.errorbar(ST, y + h / 2, xerr=ST_conf, fmt="none", capsize=3)
    ax.errorbar(S1, y - h / 2, xerr=S1_conf, fmt="none", capsize=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol sensitivity index")
    ax.set_title(f"Sobol sensitivity, W*_base = {W_base / G:.3f} kg")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("\nQBiT SOBOL SENSITIVITY SCREENING")
    print("=" * 70)
    print(f"Mission    : payload={PAYLOAD_KG:g} kg, range={RANGE_M / 1e3:.0f} km, n_c={N_C}")
    print("Evaluator  : re-optimized W_total using OpenMDAO/SLSQP")
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

    results = run_sobol(W_base)

    print_sobol_table(results, W_base)

    if not args.no_plot:
        plot_sobol(results, W_base)

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Baseline time : {baseline_time:.1f} s")
    print(f"Total runtime : {total_time:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()