"""UQ evaluation script for a deterministic design point.

This uses the existing `RobustOptimizer` and `inner_solve_for_Wtotal`
from `run_qbit_robust.py` to perform a high-fidelity Monte-Carlo / LHS
propagation on a fixed design `x_det` without running a new optimization.

Default: `n_mc=2000` (can be overridden via `--n-mc`). For quick tests use
`--quick` which sets `n_mc=200`.
"""
from __future__ import annotations
import sys
import os
import argparse
import numpy as np

# ensure sizing_openmdao is importable
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_qbit_robust import RobustOptimizer, SizingResult, inner_solve_for_Wtotal, sample_t_hover
import matplotlib.pyplot as plt
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='UQ evaluate deterministic design using RobustOptimizer internals')
    p.add_argument('--n-mc', type=int, default=2000, help='Number of MC/LHS samples (default: 2000)')
    p.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for inner solves (joblib)')
    p.add_argument('--seed', type=int, default=123, help='RNG seed for reproducibility')
    p.add_argument('--quick', action='store_true', help='Quick test: override n_mc to 200')
    args = p.parse_args()

    n_mc = 200 if args.quick else args.n_mc

    # Deterministic design point provided by user (from run_qbit.py result)
    # Order expected by inner_solve_for_Wtotal: [V_inf, r, J, S_w]
    x_det = [29.25, 0.2717, 1.300, 0.2457]
    det_W_input = 70.1  # N, for optional deterministic marker on MTOM histogram
    det_cl_input = 0.5505  # for optional deterministic marker on cruise CL plot
    # Mission configuration (match deterministic run)
    payload_kg = 3.0
    range_m = 15000.0
    n_c = 2

    print(f'Running UQ evaluation at deterministic design x_det={x_det} with n_mc={n_mc}, n_jobs={args.n_jobs}')

    # instantiate optimizer as a UQ engine (do not call .run())
    uq = RobustOptimizer(payload_kg=payload_kg, range_m=range_m, n_c=n_c,
                         n_mc=n_mc, seed=args.seed, n_jobs=args.n_jobs)

    # Pre-generate common samples (CRN) and attach to object
    uq.mc_samples = sample_t_hover(uq.n_mc, uq.mean_t, uq.std_t, uq.shift_t,
                                   seed=uq.seed, method=uq.sampling_method)

    # Try using the internal _mc_stats (fast path). If it returns None
    # (indicating some NaNs/failures), run a per-sample evaluation (with
    # optional joblib parallelism) to compute failure rate and collect valid results.
    stats = uq._mc_stats(x_det)

    if stats is not None:
        results = stats['results']
        fail_count = sum(1 for r in results if r is None)
        fail_rate = 100.0 * fail_count / len(results)
        valid_results = [r for r in results if isinstance(r, dict)]
    else:
        # Detailed per-sample fallback to collect failures and valid results
        samples = uq.mc_samples
        per_results = []
        # try joblib if user requested parallel jobs
        use_joblib = False
        if uq.n_jobs is not None and uq.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                use_joblib = True
            except Exception:
                print('joblib not available; falling back to sequential evaluation')

        if use_joblib:
            from joblib import Parallel, delayed
            per_results = Parallel(n_jobs=uq.n_jobs, prefer='processes')(
                delayed(inner_solve_for_Wtotal)(t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(x_det))
                for t in samples
            )
        else:
            for t in samples:
                r = inner_solve_for_Wtotal(t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(x_det))
                per_results.append(r if isinstance(r, dict) else None)

        results = per_results
        fail_count = sum(1 for r in results if r is None)
        fail_rate = 100.0 * fail_count / len(results)
        valid_results = [r for r in results if isinstance(r, dict)]
        if len(valid_results) == 0:
            print('All samples failed to converge; cannot compute statistics.')
            return 1

    # Compute MTOM statistics from valid results
    Wtot = np.array([p['W_total'] for p in valid_results])
    meanW = float(np.mean(Wtot))
    stdW = float(np.std(Wtot, ddof=0))
    p_lo, p_hi = np.percentile(Wtot / 9.80665, [2.5, 97.5])  # convert to kg for PI

    # Build SizingResult summary using mean values
    keys = ['W_battery','W_empty','P_hover','P_cruise','V_inf','r','J','S_w','E_req','disk_loading','blade_loading','cruise_CL','weight_residual']
    mean_res = {}
    for k in keys:
        vals = np.array([p[k] for p in valid_results])
        mean_res[k] = float(np.mean(vals))

    # compute geometry helpers (AR_FIXED is used by SizingResult.summary in original module)
    try:
        from qbit.constants import AR_FIXED
    except Exception:
        AR_FIXED = float('nan')

    b = float(np.sqrt(AR_FIXED * mean_res.get('S_w', float('nan')))) if not np.isnan(AR_FIXED) else float('nan')
    chord = mean_res.get('S_w', float('nan')) / b if b and not np.isnan(b) else float('nan')

    mean_result = SizingResult(
        W_total=meanW, W_battery=mean_res['W_battery'], W_empty=mean_res['W_empty'],
        P_hover=mean_res['P_hover'], P_cruise=mean_res['P_cruise'],
        V_inf=mean_res['V_inf'], r=mean_res['r'], J=mean_res['J'], S_w=mean_res['S_w'],
        b=b, chord=chord, E_req=mean_res['E_req'], converged=True,
        disk_loading=mean_res['disk_loading'], blade_loading=mean_res['blade_loading'], cruise_CL=mean_res['cruise_CL'], weight_residual=mean_res['weight_residual']
    )

    print('\n--- UQ Evaluation Results ---')
    print(f'Samples run      : {len(results)}')
    print(f'Failures         : {int(fail_count)} ({fail_rate:.2f} % )')
    print(f'MTOM mean (N)    : {meanW:.2f} N')
    print(f'MTOM std  (N)    : {stdW:.2f} N')
    print(f'MTOM mean (kg)   : {meanW/9.80665:.3f} kg')
    print(f'95% PI (kg)      : [{p_lo:.3f}, {p_hi:.3f}]')
    print('\nMean sizing summary:')
    print(mean_result.summary())

    # Compare against a nominal value if user wants (example 68.5 N nominal from prompt)
    nominal_N = 68.5
    growth_pct = 100.0 * (meanW - nominal_N) / nominal_N
    print(f'\nWeight growth vs nominal {nominal_N:.1f} N : {growth_pct:.2f} %')

    # --- Additional diagnostics & plots ---
    out_dir = Path(HERE) / 'uq_outputs'
    out_dir.mkdir(exist_ok=True)

    # Deterministic evaluation at mean hover time (for comparison marker)
    det_res = inner_solve_for_Wtotal(uq.mean_t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(x_det))
    det_W = None
    det_cl = None
    if isinstance(det_res, dict):
        det_W = float(det_res['W_total'])
        det_cl = float(det_res['cruise_CL'])

    # MTOM histogram (in kg) with mean, 95% PI and deterministic marker
    # --- MTOM histogram (in kg) ---
    W_kg = Wtot / 9.80665
    
    # Set your desired fixed scale for MTOM (kg) here
    mtom_min, mtom_max = 6, 9 

    fig1, ax1 = plt.subplots(figsize=(6,4))
    # Using the requested blue color
    ax1.hist(W_kg, bins=40, color='#4c72b0', edgecolor='k', alpha=0.8, label='Samples')
    
    ax1.axvline(np.mean(W_kg), color='red', linestyle='--', label=f'Mean {np.mean(W_kg):.3f} kg')
    ax1.axvline(np.percentile(W_kg, 2.5), color='gray', linestyle=':', label='95% PI')
    ax1.axvline(np.percentile(W_kg, 97.5), color='gray', linestyle=':')
    
    if det_W_input is not None:
        ax1.axvline(det_W_input/9.80665, color='purple', linestyle='-.', label=f'Deterministic {det_W_input/9.80665:.3f} kg')

    # --- FIX X-AXIS SCALE HERE ---
    ax1.set_xlim(mtom_min, mtom_max)

    ax1.set_xlabel('MTOM (kg)')
    ax1.set_ylabel('Count')
    ax1.set_title('MTOM distribution (UQ)')
    ax1.legend()
    
    p1 = out_dir / 'uq_mtom_hist.png'
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f'Saved MTOM histogram to {p1}')

    # --- Cruise CL PDF / density and violation percent ---
    cl_arr = np.array([p['cruise_CL'] for p in valid_results])
    
    # Define fixed limits
    x_min, x_max = 0.5, 0.8

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(cl_arr)
        # Generate points across the fixed scale for a smooth KDE line
        xs = np.linspace(x_min, x_max, 300)
        ys = kde(xs)
        use_kde = True
    except Exception:
        xs = None
        ys = None
        use_kde = False

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(cl_arr, bins=60, density=True, color='#55a868', alpha=0.6, edgecolor='k', label='Samples')
    
    if use_kde:
        ax2.plot(xs, ys, color='k', lw=1.2, label='KDE')

    # Mark CL limit
    try:
        from qbit.constants import CL_MAX
    except Exception:
        CL_MAX = float('nan')
    if not math.isnan(CL_MAX):
        ax2.axvline(CL_MAX, color='red', linestyle='--', lw=2, label=f'Limit CL={CL_MAX:.2f}')

    # Deterministic cruise CL marker
    if det_cl_input is not None:
        ax2.axvline(det_cl_input, color='purple', linestyle='-.', label=f'Deterministic CL={det_cl_input:.3f}')

    # --- FIX X-AXIS SCALE HERE ---
    ax2.set_xlim(x_min, x_max) 
    
    ax2.set_xlabel('Cruise CL')
    ax2.set_ylabel('Density')
    ax2.set_title('Cruise CL distribution (UQ)')
    ax2.legend()
    
    p2 = out_dir / 'uq_cruiseCL_pdf.png'
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f'Saved Cruise CL pdf to {p2}')

    # Violation percent
    if not math.isnan(CL_MAX):
        viol = np.sum(cl_arr > CL_MAX)
        viol_pct = 100.0 * viol / len(cl_arr)
        print(f'Cruise CL violations : {viol} / {len(cl_arr)} ({viol_pct:.3f} % )')
    else:
        print('CL_MAX not available from qbit.constants; cannot compute violation percent')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
