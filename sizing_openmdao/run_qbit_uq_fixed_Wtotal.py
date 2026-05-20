"""UQ evaluation with fixed `W_total` and fixed design `x_det`.

This script keeps `W_total` fixed (optionally provided) and evaluates the
required mission energy `E_req` across hover-time uncertainty samples. It
reports the percent of samples for which `E_req_sample > E_req_det` (simple
comparison) and the percent for which `E_req_sample` exceeds the available
energy implied by the fixed battery weight (more realistic infeasibility).

Usage: python uq_fixed_Wtotal.py [--n-mc N] [--W-total N] [--quick]
"""
from __future__ import annotations
import sys
import os
import argparse
import numpy as np
import math
import traceback

# ensure sizing_openmdao is importable
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_qbit_robust import RobustOptimizer, inner_solve_for_Wtotal, sample_t_hover, SizingResult
from qbit.constants import G, BATTERY_DENSITY, BATTERY_EFF
from pathlib import Path


def eval_with_fixed_W_from_cached_prob(prob, t_hover_sample: float, W_fixed: float,
                                       V_inf: float, r: float, J: float, S_w: float):
    """Run the already-setup `prob` with a fixed `W_total` and hover time.
    Returns a dict of outputs similar to `inner_solve_for_Wtotal` or None on failure.
    """
    # set global hover time used by model components (mirrors inner_solve_for_Wtotal)
    from qbit.components import sizing_comps as sc
    orig_T = getattr(sc, 'T_HOVER', None)
    sc.T_HOVER = float(t_hover_sample)
    try:
        prob.set_val('W_total', float(W_fixed))
        prob.set_val('V_inf', float(V_inf))
        prob.set_val('r', float(r))
        prob.set_val('J', float(J))
        prob.set_val('S_w', float(S_w))
        prob.run_model()
        out = {
            'W_total': float(prob.get_val('W_total')[0]),
            'W_battery': float(prob.get_val('W_battery')[0]),
            'W_empty': float(prob.get_val('W_empty')[0]),
            'P_hover': float(prob.get_val('P_hover')[0]),
            'P_cruise': float(prob.get_val('P_cruise')[0]),
            'V_inf': float(prob.get_val('V_inf')[0]),
            'r': float(prob.get_val('r')[0]),
            'J': float(prob.get_val('J')[0]),
            'S_w': float(prob.get_val('S_w')[0]),
            'E_req': float(prob.get_val('E_req')[0]),
            'disk_loading': float(prob.get_val('disk_loading')[0]),
            'blade_loading': float(prob.get_val('blade_loading')[0]),
            'cruise_CL': float(prob.get_val('cruise_CL')[0]),
            'weight_residual': float(prob.get_val('weight_residual')[0]),
        }
        return out
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if orig_T is None:
            delattr(sc, 'T_HOVER')
        else:
            sc.T_HOVER = orig_T


def main():
    p = argparse.ArgumentParser(description='UQ evaluate E_req with fixed W_total and fixed design')
    p.add_argument('--n-mc', type=int, default=2000, help='Number of MC/LHS samples (default: 2000)')
    p.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs for evaluation (optional)')
    p.add_argument('--seed', type=int, default=123, help='RNG seed')
    p.add_argument('--quick', action='store_true', help='Quick test: override n_mc to 200')
    p.add_argument('--W-total', type=float, default=None, help='Fixed W_total in N (if omitted, use deterministic value)')
    p.add_argument('--x-det', nargs=4, type=float, default=[29.79942897, 0.26030369, 1.3, 0.23397521],
                   help='Deterministic design vector: V_inf r J S_w')
    args = p.parse_args()

    n_mc = 200 if args.quick else args.n_mc
    V_inf, r, J, S_w = args.x_det

    payload_kg = 3.0
    range_m = 15000.0
    n_c = 2

    print(f'Fixed-W UQ: x_det={args.x_det}, n_mc={n_mc}, seed={args.seed}')

    # Instantiate a RobustOptimizer only to access its sampling defaults
    uq = RobustOptimizer(payload_kg=payload_kg, range_m=range_m, n_c=n_c,
                         n_mc=n_mc, seed=args.seed, n_jobs=args.n_jobs)

    # Pre-generate MC samples (CRN)
    samples = sample_t_hover(n_mc, uq.mean_t, uq.std_t, uq.shift_t, seed=uq.seed, method=uq.sampling_method)

    # Deterministic reference: run inner solve at mean hover time to get E_req_det
    print('Computing deterministic reference (inner solve at mean hover time)...')
    det = inner_solve_for_Wtotal(uq.mean_t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(args.x_det))
    if not isinstance(det, dict):
        print('Deterministic inner solve failed; aborting.')
        return 1

    E_req_det = float(det['E_req'])
    # Determine fixed W_total: user-provided or deterministic
    W_fixed = float(args.W_total) if args.W_total is not None else float(det['W_total'])
    print(f'Deterministic E_req: {E_req_det:.3f} J; using W_fixed = {W_fixed:.3f} N')

    # Retrieve (or create) the cached Problem that inner_solve_for_Wtotal created
    # Key must match inner_solve_for_Wtotal's cache key construction
    key = (float(V_inf), float(r), float(J), float(S_w), float(payload_kg), float(range_m), int(n_c))
    prob = None
    if hasattr(inner_solve_for_Wtotal, '_prob_cache'):
        prob = inner_solve_for_Wtotal._prob_cache.get(key)

    # If cache not available, attempt a single inner call to populate it
    if prob is None:
        _ = inner_solve_for_Wtotal(uq.mean_t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(args.x_det))
        if hasattr(inner_solve_for_Wtotal, '_prob_cache'):
            prob = inner_solve_for_Wtotal._prob_cache.get(key)

    # If still no prob, we will build per-sample problems (fall back)
    use_cached = prob is not None
    if use_cached:
        print('Using cached Problem for fast fixed-W evaluation.')
    else:
        print('Cached Problem not available; will setup a Problem per sample (slower).')

    results = []
    # try parallel evaluation with joblib if requested
    use_joblib = False
    if uq.n_jobs is not None and uq.n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            use_joblib = True
        except Exception:
            print('joblib not available; falling back to sequential evaluation')

    if use_joblib:
        from joblib import Parallel, delayed
        def _eval_t(t):
            if use_cached:
                return eval_with_fixed_W_from_cached_prob(prob, t, W_fixed, V_inf, r, J, S_w)
            else:
                return inner_solve_for_Wtotal(t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(args.x_det))

        res_list = Parallel(n_jobs=uq.n_jobs, prefer='processes')(delayed(_eval_t)(t) for t in samples)
        results = [r if isinstance(r, dict) else None for r in res_list]
    else:
        for t in samples:
            if use_cached:
                res = eval_with_fixed_W_from_cached_prob(prob, t, W_fixed, V_inf, r, J, S_w)
            else:
                # fallback: run inner_solve_for_Wtotal but that will attempt root-finding
                res = inner_solve_for_Wtotal(t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(args.x_det))
            results.append(res if isinstance(res, dict) else None)

    valid = [r for r in results if isinstance(r, dict) and not math.isnan(r.get('E_req', float('nan')))]
    if len(valid) == 0:
        print('No valid sample evaluations; aborting.')
        return 1

    E_arr = np.array([float(p['E_req']) for p in valid])

    # Simple percent where sample E_req exceeds deterministic E_req
    exceed_det = np.sum(E_arr > E_req_det)
    pct_exceed_det = 100.0 * exceed_det / len(E_arr)

    # More realistic infeasibility: compare sample E_req to available energy implied by fixed battery mass
    # Compute battery mass/weight under W_fixed by running a deterministic fixed-W model (if cached)
    if use_cached:
        det_fixed = eval_with_fixed_W_from_cached_prob(prob, uq.mean_t, W_fixed, V_inf, r, J, S_w)
    else:
        det_fixed = inner_solve_for_Wtotal(uq.mean_t, uq.payload_kg, uq.range_m, uq.n_c, design_vars=tuple(args.x_det))

    if not isinstance(det_fixed, dict):
        print('Could not compute battery for fixed W_total; reporting only simple exceedance metric.')
        print(f'Percent samples with E_req > E_req_det: {pct_exceed_det:.3f} %')
        return 0

    W_bat_fixed = float(det_fixed['W_battery'])
    # available energy [J]
    E_avail_fixed = (W_bat_fixed / G) * BATTERY_DENSITY * BATTERY_EFF * 3600.0

    # debug: show E_arr stats and compute exceedance two ways
    print(f'DEBUG: E_avail_fixed={E_avail_fixed:.1f} J from W_battery={W_bat_fixed:.3f} N')
    print(f'DEBUG: E_arr min={E_arr.min():.1f}, mean={E_arr.mean():.1f}, max={E_arr.max():.1f}')
    mask = E_arr > E_avail_fixed
    exceed_avail = int(np.sum(mask))
    pct_exceed_avail = 100.0 * exceed_avail / len(E_arr)
    # also compute via python loop for sanity
    exceed_loop = sum(1 for v in E_arr if v > E_avail_fixed)
    if exceed_loop != exceed_avail:
        print(f'DEBUG: mismatch counts: vectorized={exceed_avail} loop={exceed_loop}')
    if exceed_avail > 0:
        print(f'DEBUG: first exceed indices: {np.where(mask)[0][:10].tolist()}')

    print('\n--- Fixed-W UQ Results ---')
    print(f'Samples evaluated       : {len(results)}')
    print(f'Valid samples           : {len(valid)}')
    print(f'Percent E_req > E_req_det : {pct_exceed_det:.3f} % ({exceed_det}/{len(valid)})')
    print(f'Fixed battery energy (J): {E_avail_fixed:.1f} J (from W_battery={W_bat_fixed:.3f} N)')
    print(f'Percent E_req > E_avail_fixed : {pct_exceed_avail:.3f} % ({exceed_avail}/{len(valid)})')

    # Save simple CSV of sample t_hover and E_req for further analysis
    out_dir = Path(HERE) / 'uq_outputs'
    out_dir.mkdir(exist_ok=True)
    csv_p = out_dir / 'fixedW_Ereq_samples.csv'
    # also save exceeds_avail flag aligned to valid samples
    # map samples -> valid indices: keep only samples corresponding to valid entries
    valid_samples = []
    for i, r in enumerate(results):
        if isinstance(r, dict) and not math.isnan(r.get('E_req', float('nan'))):
            valid_samples.append(samples[i])
    arr_to_save = np.column_stack((np.array(valid_samples), E_arr, mask.astype(int)))
    np.savetxt(csv_p, arr_to_save, delimiter=',', header='t_hover,E_req_J,exceeds_avail', comments='')
    print(f'Saved sample E_req to {csv_p}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
