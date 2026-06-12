"""
run_qbit_robust.py - Monte Carlo robust sizing optimization for QBiT.

This is the direct Monte Carlo counterpart of run_qbit_uqpce.py.

Workflow:
1. Draw fixed hover-time samples from a shifted lognormal distribution.
2. For each proposed design, evaluate QBiT at all hover-time samples.
3. Estimate mean MTOM, sample 97.5th percentile MTOM, and mean constraints.
4. Minimize the sample 97.5th percentile of MTOM.
5. Validate the optimized design with a larger Monte Carlo sample.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, qmc
import openmdao.api as om

from run_qbit import evaluate_qbit_at_design
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (
    G,
    V_INF_BOUNDS,
    R_BOUNDS,
    J_BOUNDS,
    S_W_BOUNDS,
    DL_MAX,
    BL_MAX,
    CL_MAX,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

SEED = 123

# Mission definition
PAYLOAD_KG = 3.0
RANGE_M = 15_000.0
N_CUSTOMERS = 2

# Hover-time requirement uncertainty
T_HOVER_SHIFT = 25.0
T_HOVER_MEAN = 55.0
T_HOVER_STD = 18.0

# Monte Carlo settings
MC_OPT_SAMPLES = 1000
MC_VALIDATION_SAMPLES = 1000
SAMPLING_METHOD = "lhs"  # "lhs" or "mcs"

# Optimizer settings
OPT_MAXITER = 200
OPT_FTOL = 1e-6

# Initial design
X0 = [28.70, 0.2678, 1.3, 0.2296]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def get_shifted_lognormal_dist(target_mean: float, target_std: float, shift: float):
    """
    Return a shifted lognormal distribution with prescribed mean and std.

    t_hover = shift + lognormal(...)
    """

    mean_unshifted = target_mean - shift
    var_unshifted = target_std**2

    sigma_sq = math.log(var_unshifted / mean_unshifted**2 + 1.0)
    sigma = math.sqrt(sigma_sq)
    scale = mean_unshifted / math.sqrt(var_unshifted / mean_unshifted**2 + 1.0)

    return lognorm(s=sigma, loc=shift, scale=scale)


def sample_t_hover(
    n_samples: int,
    mean: float = T_HOVER_MEAN,
    std: float = T_HOVER_STD,
    shift: float = T_HOVER_SHIFT,
    seed: int | None = None,
    method: str = SAMPLING_METHOD,
) -> np.ndarray:
    """
    Generate hover-time samples.

    LHS is the default because it gives smoother sample statistics than plain
    random Monte Carlo for the same sample count.
    """

    dist = get_shifted_lognormal_dist(mean, std, shift)

    if method.lower() in ("lhs", "latin", "latin_hypercube", "latin-hypercube"):
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        u = sampler.random(n=n_samples).ravel()
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        return dist.ppf(u)

    rng = np.random.default_rng(seed)
    return dist.rvs(size=n_samples, random_state=rng)


# ---------------------------------------------------------------------------
# Fixed-design evaluation
# ---------------------------------------------------------------------------

def build_inner_problem(payload_kg: float, range_m: float, n_c: int) -> om.Problem:
    """
    Build one reusable QBiT problem.

    The design variables and hover time are changed during evaluation.
    """

    prob = om.Problem(reports=None)
    prob.model = build_qbit_model(payload_kg, range_m, n_c)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    prob.model.set_input_defaults("V_inf", val=35.0, units="m/s")
    prob.model.set_input_defaults("r", val=0.20, units="m")
    prob.model.set_input_defaults("J", val=1.3)
    prob.model.set_input_defaults("S_w", val=0.30, units="m**2")

    prob.setup()
    prob.run_model()

    return prob


def evaluate_design_samples(
    x: Sequence[float],
    samples: np.ndarray,
    payload_kg: float,
    range_m: float,
    n_c: int,
    strict: bool = True,
) -> list[dict] | None:
    """
    Evaluate one design for all hover-time samples.

    If strict=True, any failed inner solve invalidates the design.
    This is useful during optimization.
    """

    dvars = tuple(float(v) for v in x)
    prob = build_inner_problem(payload_kg, range_m, n_c)

    results = []
    n_failed = 0

    for t_hover in samples:
        res = evaluate_qbit_at_design(
            prob=prob,
            t_hover=float(t_hover),
            dvars=dvars,
        )

        if res is None:
            n_failed += 1
            if strict:
                return None
        else:
            results.append(res)

    if not results:
        return None

    if n_failed:
        print(f"\n  MC warning: {n_failed}/{len(samples)} samples failed")

    return results


# ---------------------------------------------------------------------------
# Monte Carlo optimizer
# ---------------------------------------------------------------------------

@dataclass
class RobustOptimizer:
    payload_kg: float
    range_m: float
    n_c: int = 1
    n_mc: int = MC_OPT_SAMPLES
    mean_t: float = T_HOVER_MEAN
    std_t: float = T_HOVER_STD
    shift_t: float = T_HOVER_SHIFT
    seed: int | None = SEED
    mc_samples: np.ndarray | None = None
    sampling_method: str = SAMPLING_METHOD
    maxiter: int = OPT_MAXITER

    _obj_calls: int = 0
    eval_times: list | None = None

    def _mc_stats(self, x: Sequence[float], strict: bool = True):
        """
        Evaluate one design over the MC sample set and return sample statistics.

        This method name is kept so the UQPCE script can still use it for
        Monte Carlo validation.
        """

        if self.mc_samples is None:
            samples = sample_t_hover(
                self.n_mc,
                self.mean_t,
                self.std_t,
                self.shift_t,
                seed=self.seed,
                method=self.sampling_method,
            )
        else:
            samples = self.mc_samples

        results = evaluate_design_samples(
            x=x,
            samples=samples,
            payload_kg=self.payload_kg,
            range_m=self.range_m,
            n_c=self.n_c,
            strict=strict,
        )

        if results is None:
            return None

        W_arr = np.array([res["W_total"] for res in results])

        mean_res = {}
        for key in results[0].keys():
            mean_res[key] = float(np.mean([res[key] for res in results]))

        meanW = float(np.mean(W_arr))
        stdW = float(np.std(W_arr, ddof=0))
        p97_5_sample = float(np.percentile(W_arr, 97.5))

        return {
            "meanW": meanW,
            "stdW": stdW,
            "p97_5W": p97_5_sample,
            "p97_5_sample": p97_5_sample,
            "mean_res": mean_res,
            "results": results,
            "W_samples": W_arr,
        }

    def objective(self, x: Sequence[float]) -> float:
        """
        Objective: sample-based 97.5th percentile of MTOM.

        This is intentionally noisy compared with UQPCE, but it is the direct
        Monte Carlo analogue of an upper-tail robust objective.
        """

        start = time.time()
        stats = self._mc_stats(x, strict=True)
        elapsed = time.time() - start

        if self.eval_times is None:
            self.eval_times = []

        self.eval_times.append(elapsed)
        self._obj_calls += 1

        if stats is None:
            print(f"\rEval {self._obj_calls:03d}: FAILED", end="", flush=True)
            return 1e6

        U = float(stats["p97_5_sample"])

        print(
            f"\rEval {self._obj_calls:03d}: "
            f"V={x[0]:.2f}, r={x[1]:.4f}, J={x[2]:.3f}, S_w={x[3]:.4f} | "
            f"mean={stats['meanW'] / G:.3f} kg, "
            f"robust={U / G:.3f} kg",
            end="",
            flush=True,
        )

        return U

    def run(self, x0: Sequence[float] | None = None, method: str = "SLSQP"):
        """
        Run the outer robust optimization.

        Common random numbers are used: every design is evaluated with the
        same MC sample set. This reduces random noise in objective differences.
        """

        if x0 is None:
            x0 = X0

        self.mc_samples = sample_t_hover(
            self.n_mc,
            self.mean_t,
            self.std_t,
            self.shift_t,
            seed=self.seed,
            method=self.sampling_method,
        )

        bounds = [V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS]

        def constr_cruise_CL(x):
            stats = self._mc_stats(x, strict=True)
            if stats is None:
                return -1e6
            return float(CL_MAX - stats["mean_res"]["cruise_CL"])

        def constr_disk_loading(x):
            stats = self._mc_stats(x, strict=True)
            if stats is None:
                return -1e6
            return float(DL_MAX - stats["mean_res"]["disk_loading"])

        def constr_blade_loading(x):
            stats = self._mc_stats(x, strict=True)
            if stats is None:
                return -1e6
            return float(BL_MAX - stats["mean_res"]["blade_loading"])

        cons = [
            {"type": "ineq", "fun": constr_cruise_CL},
            {"type": "ineq", "fun": constr_disk_loading},
            {"type": "ineq", "fun": constr_blade_loading},
        ]

        self._obj_calls = 0
        self.eval_times = []

        res = minimize(
            self.objective,
            np.array(x0, dtype=float),
            method=method,
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": self.maxiter,
                "ftol": OPT_FTOL,
                "disp": True,
            },
        )

        return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_initial_mc_check(opt: RobustOptimizer, x0: Sequence[float]) -> None:
    stats = opt._mc_stats(x0, strict=True)

    if stats is None:
        print("\nInitial design MC check failed.")
        return

    print("\nInitial design MC check:")
    print(f"  Mean MTOM   : {stats['meanW'] / G:.3f} kg")
    print(f"  Std MTOM    : {stats['stdW']:.2f} N")
    print(f"  Robust MTOM : {stats['p97_5_sample'] / G:.3f} kg")


def print_results(res, stats: dict, opt_time: float) -> None:
    V_inf, r, J, S_w = res.x
    x_opt = [float(V_inf), float(r), float(J), float(S_w)]

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"  Converged      : {res.success}")
    print(f"  V_inf          : {V_inf:.2f} m/s")
    print(f"  Rotor radius r : {r:.4f} m")
    print(f"  Prop adv. J    : {J:.3f}")
    print(f"  Wing area S_w  : {S_w:.4f} m²")

    print()
    print(f"  Design vector  : {x_opt}")

    print()
    print(f"  Mean MTOM      : {stats['meanW'] / G:.3f} kg")
    print(f"  MTOM variance  : {stats['stdW']**2:.4f} N²")
    print(f"  Robust MTOM    : {stats['p97_5_sample'] / G:.3f} kg")

    print()
    print("  Constraint values:")
    print(f"    cruise_CL_mean    : {stats['mean_res']['cruise_CL']:.4f} ≤ {CL_MAX}")
    print(f"    disk_loading_mean : {stats['mean_res']['disk_loading']:.2f} N/m² ≤ {DL_MAX}")
    print(f"    blade_loading_mean: {stats['mean_res']['blade_loading']:.4f} ≤ {BL_MAX}")

    print()
    print(f"  Optimization iterations: {res.nit}")
    print(f"  Function evaluations   : {res.nfev}")
    print(f"  Optimization time      : {opt_time:.1f} s")

    print("=" * 60)


def validate_final_design(x_opt: Sequence[float]) -> dict | None:
    opt_large = RobustOptimizer(
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS,
        n_mc=MC_VALIDATION_SAMPLES,
        mean_t=T_HOVER_MEAN,
        std_t=T_HOVER_STD,
        shift_t=T_HOVER_SHIFT,
        seed=SEED + 1,
        sampling_method=SAMPLING_METHOD,
    )

    opt_large.mc_samples = sample_t_hover(
        MC_VALIDATION_SAMPLES,
        T_HOVER_MEAN,
        T_HOVER_STD,
        T_HOVER_SHIFT,
        seed=SEED + 1,
        method=SAMPLING_METHOD,
    )

    stats = opt_large._mc_stats(x_opt, strict=False)

    if stats is None:
        print("\nMonte Carlo validation failed: no valid samples.")
        return None

    print(f"\nValidation samples solved: {len(stats['results'])}/{MC_VALIDATION_SAMPLES}")
    return stats


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    print("=" * 70)
    print("MONTE CARLO ROBUST DESIGN OPTIMIZATION")
    print("=" * 70)

    opt = RobustOptimizer(
        payload_kg=PAYLOAD_KG,
        range_m=RANGE_M,
        n_c=N_CUSTOMERS,
        n_mc=MC_OPT_SAMPLES,
        mean_t=T_HOVER_MEAN,
        std_t=T_HOVER_STD,
        shift_t=T_HOVER_SHIFT,
        seed=SEED,
        sampling_method=SAMPLING_METHOD,
        maxiter=OPT_MAXITER,
    )

    print(f"\nMC samples per design : {MC_OPT_SAMPLES}")
    print(f"Validation samples    : {MC_VALIDATION_SAMPLES}")
    print(f"Sampling method       : {SAMPLING_METHOD}")
    print("Objective             : sample 97.5th percentile of MTOM")
    print("Constraints           : mean CL, disk loading, blade loading")

    print("\n[1/4] Checking initial design...")
    opt.mc_samples = sample_t_hover(
        MC_OPT_SAMPLES,
        T_HOVER_MEAN,
        T_HOVER_STD,
        T_HOVER_SHIFT,
        seed=SEED,
        method=SAMPLING_METHOD,
    )
    print_initial_mc_check(opt, X0)

    print("\n[2/4] Running robust optimization...")
    opt_start = time.time()
    res = opt.run(x0=X0)
    opt_time = time.time() - opt_start

    print("\n[3/4] Validating final design with larger MC sample...")
    val_start = time.time()
    stats = validate_final_design(res.x)
    val_time = time.time() - val_start

    if stats is not None:
        print_results(res, stats, opt_time)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Optimization time : {opt_time:.1f} s")
    print(f"Validation time   : {val_time:.1f} s")
    print(f"Total runtime     : {total_time:.1f} s")
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    main()