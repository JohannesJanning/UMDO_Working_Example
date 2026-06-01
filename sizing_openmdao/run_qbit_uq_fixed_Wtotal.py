"""UQ evaluation with fixed `W_total` and fixed design `x_det` over a specific hover range.

This script sweeps hover times from 55s to 101s and prints the resulting mission energy
in Watt-hours (Wh) and power consumption (in Watts) directly to the terminal.
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

from run_qbit_MCS import RobustOptimizer, inner_solve_for_Wtotal, SizingResult
from qbit.constants import G, BATTERY_DENSITY, BATTERY_EFF
from pathlib import Path


def eval_with_fixed_W_from_cached_prob(prob, t_hover_sample: float, W_fixed: float,
                                       V_inf: float, r: float, J: float, S_w: float):
    """Run the already-setup `prob` with a fixed `W_total` and hover time."""
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
    p = argparse.ArgumentParser(description='Sweep hover time and calculate power consumption in Watts')
    p.add_argument('--W-total', type=float, default=6.981*9.81, help='Fixed W_total in N')
    p.add_argument('--x-det', nargs=4, type=float, default=[31.36, 0.2227, 1.3, 0.1895],
                   help='Deterministic design vector: V_inf r J S_w')
    args = p.parse_args()

    V_inf, r, J, S_w = args.x_det
    payload_kg = 3.0
    range_m = 15000.0
    n_c = 2

    # Calculate constant cruise time (t = distance / speed) to get total mission time
    t_cruise = range_m / V_inf

    # Generate a clean sweep from 55s to 101s in increments of 5 seconds (including 101s exactly)
    samples = list(np.arange(55.0, 101.0, 5.0)) + [101.0]

    # Force the baseline reference evaluation to occur at the design-point optimization value (101s)
    design_hover_time = 55.0

    print(f'Computing baseline reference configuration sized at {design_hover_time}s hover...')
    det = inner_solve_for_Wtotal(design_hover_time, payload_kg, range_m, n_c, design_vars=tuple(args.x_det))
    if not isinstance(det, dict):
        print('Baseline initialization failed; aborting.')
        return 1

    W_fixed = float(args.W_total) if args.W_total is not None else float(det['W_total'])

    # Retrieve cached Problem instance
    key = (float(V_inf), float(r), float(J), float(S_w), float(payload_kg), float(range_m), int(n_c))
    prob = None
    if hasattr(inner_solve_for_Wtotal, '_prob_cache'):
        prob = inner_solve_for_Wtotal._prob_cache.get(key)

    if prob is None:
        _ = inner_solve_for_Wtotal(design_hover_time, payload_kg, range_m, n_c, design_vars=tuple(args.x_det))
        if hasattr(inner_solve_for_Wtotal, '_prob_cache'):
            prob = inner_solve_for_Wtotal._prob_cache.get(key)

    use_cached = prob is not None
    
    # Calculate fixed energy capacity available based on the 101s optimized battery weight allocation
    if use_cached:
        det_fixed = eval_with_fixed_W_from_cached_prob(prob, design_hover_time, W_fixed, V_inf, r, J, S_w)
    else:
        det_fixed = inner_solve_for_Wtotal(design_hover_time, payload_kg, range_m, n_c, design_vars=tuple(args.x_det))

    W_bat_fixed = float(det_fixed['W_battery'])
    E_avail_fixed_J = (W_bat_fixed / G) * BATTERY_DENSITY * BATTERY_EFF * 3600.0
    E_avail_fixed_Wh = E_avail_fixed_J / 3600.0
    
    # Terminal Output Header
    print('\n' + '='*85)
    print(f" HOVER SWEEP ANALYSIS (55s to 101s) | Fixed Weight: {W_fixed:.2f} N")
    print(f" Sized Baseline Hover Time        : {design_hover_time} s")
    print(f" Available Battery Energy Cap     : {E_avail_fixed_Wh:.2f} Wh")
    print('='*85)
    print(f"{'Hover Time':<12} | {'Total Mission Energy':<22} | {'Avg Mission Power':<20} | {'Status':<15}")
    print('-'*85)

    for t in samples:
        if use_cached:
            res = eval_with_fixed_W_from_cached_prob(prob, t, W_fixed, V_inf, r, J, S_w)
        else:
            res = inner_solve_for_Wtotal(t, payload_kg, range_m, n_c, design_vars=tuple(args.x_det))
        
        if isinstance(res, dict) and not math.isnan(res.get('E_req', float('nan'))):
            E_req_J = float(res['E_req'])
            E_req_Wh = E_req_J / 3600.0
            
            # Mission total duration = current hover segment time + fixed forward flight cruise time
            t_total = t + t_cruise
            
            # Average Power Consumption (W) = Energy (J) / Total Time (s)
            P_avg_mission = E_req_J / t_total
            
            # Use a tiny buffer for floating-point precision at exactly 101s
            status = "FEASIBLE" if E_req_J <= (E_avail_fixed_J + 1e-2) else "OUT OF BATTERY"
            
            print(f"{t:<10.1f} s | {E_req_Wh:<19.2f} Wh | {P_avg_mission:<17.2f} W  | {status:<15}")
        else:
            print(f"{t:<10.1f} s | {'Simulation Failed':<22} | {'N/A':<20} | {'FAILED':<15}")
            
    print('='*85 + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())