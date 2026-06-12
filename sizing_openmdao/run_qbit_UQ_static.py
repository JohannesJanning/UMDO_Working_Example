"""
run_qbit_uq_eval.py - UQ evaluation of one fixed QBiT design.

This script does not optimize.

It takes one fixed design x_det = [V_inf, r, J, S_w], samples uncertain
hover time, evaluates the OpenMDAO QBiT model for each sample through
RobustOptimizer._mc_stats(), and reports MTOM and constraint statistics.

It uses the cleaned run_qbit_monte_carlo.py functions:
- RobustOptimizer
- sample_t_hover
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Ensure local modules are importable when running from this folder.
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_qbit_monte_carlo import RobustOptimizer, sample_t_hover
from qbit.constants import G, DL_MAX, BL_MAX, CL_MAX


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Fixed deterministic design to evaluate: [V_inf, r, J, S_w]
X_DET = [28.689745845105193, 0.26790218985570596, 1.2999999999999492, 0.229714837281008]

# Design vector for t-hover=55s deterministic MDO: X_DET = [31.3564009542092, 0.2226826129230352, 1.3, 0.189523953710358]
# Design vector for t-hover=101s deterministic MDO: X_DET = [29.33041342078393, 0.26979795053605055, 1.3, 0.24352624788169958]
# Design vector for UQPCE: [28.689745845105193, 0.26790218985570596, 1.2999999999999492, 0.229714837281008]
# Design vector for LHS: [28.699444126435683, 0.2677791035793159, 1.2999999999994207, 0.22953996469383947]

# Mission definition
PAYLOAD_KG = 3.0
RANGE_M = 15_000.0
N_CUSTOMERS = 2

# Hover-time requirement uncertainty
T_HOVER_MEAN = 55.0
T_HOVER_STD = 18.0
T_HOVER_SHIFT = 25.0

# Sampling
N_MC_DEFAULT = 2000
N_MC_QUICK = 200
SEED = 123
SAMPLING_METHOD = "lhs"  # "lhs" or "mcs"

# Confidence-ratio budget
W_CERT_KG = 7.850


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def violation_summary(values: np.ndarray, limit: float) -> tuple[int, float]:
    """Return number and percentage of samples above a constraint limit."""
    n_viol = int(np.sum(values > limit))
    pct_viol = 100.0 * n_viol / len(values)
    return n_viol, pct_viol


def print_uq_results(
    x_det: list[float],
    stats: dict,
    n_requested: int,
    runtime: float,
) -> None:
    """Print UQ results in a compact report style."""

    results = stats["results"]
    W_samples = stats["W_samples"]

    valid_count = len(results)
    fail_count = n_requested - valid_count
    fail_rate = 100.0 * fail_count / n_requested

    W_kg = W_samples / G
    p_2_5 = float(np.percentile(W_kg, 2.5))
    p_50 = float(np.percentile(W_kg, 50.0))
    p_97_5 = float(np.percentile(W_kg, 97.5))

    cruise_CL_arr = np.array([res["cruise_CL"] for res in results])
    disk_loading_arr = np.array([res["disk_loading"] for res in results])
    blade_loading_arr = np.array([res["blade_loading"] for res in results])

    cl_viol, cl_viol_pct = violation_summary(cruise_CL_arr, CL_MAX)
    dl_viol, dl_viol_pct = violation_summary(disk_loading_arr, DL_MAX)
    bl_viol, bl_viol_pct = violation_summary(blade_loading_arr, BL_MAX)

    cr_margin = W_CERT_KG - p_97_5
    cr_uncertainty = p_97_5 - p_50
    cr = cr_margin / cr_uncertainty if cr_uncertainty > 0 else float("nan")

    if cr < 0:
        cr_case = "negative margin; exceeds budget"
    elif cr < 1:
        cr_case = "small positive margin; reliability in question"
    else:
        cr_case = "large positive margin; reliable"

    V_inf, r, J, S_w = x_det

    print("\n" + "=" * 60)
    print("FIXED-DESIGN UQ EVALUATION")
    print("=" * 60)

    print(f"  V_inf          : {V_inf:.2f} m/s")
    print(f"  Rotor radius r : {r:.4f} m")
    print(f"  Prop adv. J    : {J:.3f}")
    print(f"  Wing area S_w  : {S_w:.4f} m²")

    print()
    print(f"  Samples requested : {n_requested}")
    print(f"  Samples solved    : {valid_count}")
    print(f"  Failures          : {fail_count} ({fail_rate:.2f} %)")

    print()
    print(f"  Mean MTOM      : {stats['meanW'] / G:.3f} kg")
    print(f"  Std MTOM       : {stats['stdW'] / G:.3f} kg")
    print(f"  Median MTOM    : {p_50:.3f} kg")
    print(f"  95% PI MTOM    : [{p_2_5:.3f}, {p_97_5:.3f}] kg")
    print(f"  Robust MTOM    : {p_97_5:.3f} kg")

    print()
    print("  Mean constraint values:")
    print(f"    cruise_CL_mean    : {stats['mean_res']['cruise_CL']:.4f} ≤ {CL_MAX}")
    print(f"    disk_loading_mean : {stats['mean_res']['disk_loading']:.2f} N/m² ≤ {DL_MAX}")
    print(f"    blade_loading_mean: {stats['mean_res']['blade_loading']:.4f} ≤ {BL_MAX}")

    print()
    print("  Constraint violation rates:")
    print(f"    cruise_CL     : {cl_viol}/{valid_count} ({cl_viol_pct:.2f} %)")
    print(f"    disk_loading  : {dl_viol}/{valid_count} ({dl_viol_pct:.2f} %)")
    print(f"    blade_loading : {bl_viol}/{valid_count} ({bl_viol_pct:.2f} %)")

    print()
    print("  Confidence ratio:")
    print(f"    W_cert      : {W_CERT_KG:.3f} kg")
    print(f"    Margin      : {cr_margin * 1000:+.1f} g")
    print(f"    Uncertainty : {cr_uncertainty * 1000:.1f} g")
    print(f"    CR          : {cr:+.4f}")
    print(f"    Case        : {cr_case}")

    print()
    print(f"  Runtime       : {runtime:.1f} s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed QBiT design under hover-time uncertainty."
    )
    parser.add_argument("--n-mc", type=int, default=N_MC_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--method", type=str, default=SAMPLING_METHOD, choices=["lhs", "mcs"])
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    n_mc = N_MC_QUICK if args.quick else args.n_mc

    print("=" * 70)
    print("FIXED-DESIGN MONTE CARLO / LHS UQ EVALUATION")
    print("=" * 70)
    print(f"Design x_det       : {X_DET}")
    print(f"Samples            : {n_mc}")
    print(f"Sampling method    : {args.method}")
    print(f"Seed               : {args.seed}")

    start = time.time()

    uq = RobustOptimizer(
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS,
        n_mc=n_mc,
        mean_t=T_HOVER_MEAN,
        std_t=T_HOVER_STD,
        shift_t=T_HOVER_SHIFT,
        seed=args.seed,
        sampling_method=args.method,
    )

    uq.mc_samples = sample_t_hover(
        n_mc,
        T_HOVER_MEAN,
        T_HOVER_STD,
        T_HOVER_SHIFT,
        seed=args.seed,
        method=args.method,
    )

    stats = uq._mc_stats(X_DET, strict=False)

    runtime = time.time() - start

    if stats is None:
        print("\nUQ evaluation failed: no valid samples.")
        return 1

    print_uq_results(
        x_det=X_DET,
        stats=stats,
        n_requested=n_mc,
        runtime=runtime,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())