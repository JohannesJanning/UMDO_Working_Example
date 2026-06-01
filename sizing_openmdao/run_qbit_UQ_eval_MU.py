"""UQ evaluation script for a deterministic design point — two uncertain parameters.

Propagates joint uncertainty in:
  1. T_HOVER   [s]  – hover time (shifted lognormal, mean=55 s, std=18 s)
  2. ETA_HOVER [–]  – rotor hover figure of merit (truncated normal,
                      mean=0.65, std=0.04, lo=0.50, hi=0.80)

ETA_HOVER couples to W_total through two channels:
  (a) P_hover ∝ 1/ETA_HOVER  →  E_req  →  W_battery  (energy)
  (b) P_hover ∝ 1/ETA_HOVER  →  P_inst →  W_empty    (motor sizing)
This joint interaction means safety-factor stacking is genuinely sub-optimal.

Usage:
    python run_qbit_UQ_eval_MU.py              # 2000 samples
    python run_qbit_UQ_eval_MU.py --quick      # 200 samples
    python run_qbit_UQ_eval_MU.py --n-mc 500
"""
from __future__ import annotations

import sys
import os
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_qbit_MCS_MU import (
    RobustOptimizer,
    SizingResult,
    inner_solve_for_Wtotal,
    sample_uncertain_inputs,
    eta_hover_deterministic_comparison,
    ETA_HOVER_MEAN,
    ETA_HOVER_STD,
    ETA_HOVER_LO,
    ETA_HOVER_HI,
    _T_HOVER_DIST,
    _ETA_HOVER_DIST,
)

from qbit.constants import G, DL_MAX, BL_MAX, CL_MAX, AR_FIXED


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit here
# ─────────────────────────────────────────────────────────────────────────────

# Deterministic optimum  [V_inf (m/s), r (m), J (-), S_w (m²)]
# (sized at T_HOVER=101 s, ETA_HOVER=0.572 — stacked worst case)
X_DET = [27.48, 0.2932, 1.300, 0.3139]

# Reference values from the deterministic run (for plot markers)
DET_W_N  = 70.1    # N   – deterministic MTOM
DET_CL   = 0.5505  # –   – deterministic cruise CL

# Mission
PAYLOAD_KG = 3.0
RANGE_M    = 15_000.0
N_C        = 2

# Plot axis limits
MTOM_LIM        = (6.0,  9.0)     # kg
CL_LIM          = (0.45, 0.75)    # –
DL_LIM          = (50,  150)      # N/m²
BL_LIM          = (0.012, 0.018)  # –
ETA_SCAT_LIM    = (0.50, 0.80)    # – (scatter colour axis)
T_HOV_SCAT_LIM  = (20,   120)     # s


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _violation_stats(arr: np.ndarray, limit: float, direction: str = 'upper'):
    if math.isnan(limit):
        return 0, float('nan')
    mask = arr > limit if direction == 'upper' else arr < limit
    n = int(np.sum(mask))
    return n, 100.0 * n / len(arr)


def _hist_with_kde(ax, data, color, bins=60, label='Samples'):
    ax.hist(data, bins=bins, density=True, color=color,
            alpha=0.55, edgecolor='k', linewidth=0.4, label=label)
    try:
        kde = gaussian_kde(data)
        xs  = np.linspace(data.min(), data.max(), 400)
        ax.plot(xs, kde(xs), color='k', lw=1.4, label='KDE')
        return kde
    except Exception:
        return None


def _add_limit_line(ax, limit, label, color='red'):
    if not math.isnan(limit):
        ax.axvline(limit, color=color, ls='--', lw=1.8, label=label)


def _fill_violation(ax, kde, limit, x_max, color='red', label=''):
    if kde is None or math.isnan(limit):
        return
    xs = np.linspace(limit, x_max, 200)
    ax.fill_between(xs, 0, kde(xs), color=color, alpha=0.20, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='UQ forward propagation – joint (T_HOVER, ETA_HOVER)')
    ap.add_argument('--n-mc',   type=int,  default=2000)
    ap.add_argument('--n-jobs', type=int,  default=1)
    ap.add_argument('--seed',   type=int,  default=123)
    ap.add_argument('--quick',  action='store_true', help='Use n_mc=200')
    args = ap.parse_args()

    n_mc = 200 if args.quick else args.n_mc
    seed = args.seed

    print('=' * 65)
    print('QBiT UQ Evaluation  –  joint (T_HOVER, ETA_HOVER) uncertainty')
    print('=' * 65)
    print(f'  Design x_det  : {X_DET}')
    print(f'  Mission       : {PAYLOAD_KG} kg payload, {RANGE_M/1e3:.0f} km range, {N_C} customers')
    print(f'  Samples       : {n_mc}  (seed={seed})')
    print(f'  T_HOVER       : shifted-lognormal  mean=55 s, std=18 s')
    print(f'  ETA_HOVER     : truncated-normal   mean={ETA_HOVER_MEAN:.3f}, '
          f'std={ETA_HOVER_STD:.3f}')
    print(f'  Det. ETA_HOVER: {eta_hover_deterministic_comparison:.4f}  '
          f'(2.5th pct, worst-case for det. comparison)')
    print()

    # ── generate joint LHS samples ──────────────────────────────────────────
    samples = sample_uncertain_inputs(n_mc, seed=seed, method='lhs')
    # samples[:,0] = T_HOVER [s],  samples[:,1] = ETA_HOVER [–]

    # ── run inner solves ─────────────────────────────────────────────────────
    use_joblib = args.n_jobs != 1
    if use_joblib:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            use_joblib = False
            print('joblib not available – falling back to sequential')

    def _call(row):
        return inner_solve_for_Wtotal(
            float(row[0]), float(row[1]),
            PAYLOAD_KG, RANGE_M, N_C,
            design_vars=tuple(X_DET),
        )

    if use_joblib:
        results = Parallel(n_jobs=args.n_jobs, prefer='processes')(
            delayed(_call)(samples[i]) for i in range(n_mc)
        )
    else:
        results = []
        for i, row in enumerate(samples):
            r = _call(row)
            results.append(r if isinstance(r, dict) else None)
            if (i + 1) % 50 == 0:
                print(f'  ... {i+1}/{n_mc} samples done')

    # ── filter valid ─────────────────────────────────────────────────────────
    fail_count = sum(1 for r in results if r is None)
    valid      = [r for r in results if isinstance(r, dict)]
    fail_rate  = 100.0 * fail_count / len(results)

    if not valid:
        print('ERROR: all samples failed to converge.')
        return 1

    print(f'\n  Converged: {len(valid)}/{n_mc}  (failures: {fail_count}, {fail_rate:.1f} %)')

    # ── extract arrays ───────────────────────────────────────────────────────
    W_arr   = np.array([r['W_total']          for r in valid])
    CL_arr  = np.array([r['cruise_CL']        for r in valid])
    DL_arr  = np.array([r['disk_loading']     for r in valid])
    BL_arr  = np.array([r['blade_loading']    for r in valid])
    T_arr   = np.array([r['t_hover_sample']   for r in valid])
    ETA_arr = np.array([r['eta_hover_sample'] for r in valid])

    W_kg = W_arr / G

    # ── statistics ───────────────────────────────────────────────────────────
    meanW  = float(np.mean(W_arr))
    stdW   = float(np.std(W_arr,  ddof=0))
    p2_5   = float(np.percentile(W_kg, 2.5))
    p97_5  = float(np.percentile(W_kg, 97.5))

    n_viol_cl, pct_viol_cl = _violation_stats(CL_arr, CL_MAX)
    n_viol_dl, pct_viol_dl = _violation_stats(DL_arr, DL_MAX)
    n_viol_bl, pct_viol_bl = _violation_stats(BL_arr, BL_MAX)

    keys = ['W_battery', 'W_empty', 'P_hover', 'P_cruise', 'V_inf', 'r', 'J', 'S_w',
            'E_req', 'disk_loading', 'blade_loading', 'cruise_CL', 'weight_residual']
    mrd  = {k: float(np.mean([r[k] for r in valid])) for k in keys}
    b_   = float(np.sqrt(AR_FIXED * mrd['S_w']))
    ch_  = mrd['S_w'] / b_

    sr = SizingResult(
        W_total=meanW, W_battery=mrd['W_battery'], W_empty=mrd['W_empty'],
        P_hover=mrd['P_hover'], P_cruise=mrd['P_cruise'],
        V_inf=mrd['V_inf'], r=mrd['r'], J=mrd['J'], S_w=mrd['S_w'],
        b=b_, chord=ch_, E_req=mrd['E_req'], converged=True,
        disk_loading=mrd['disk_loading'], blade_loading=mrd['blade_loading'],
        cruise_CL=mrd['cruise_CL'], weight_residual=mrd['weight_residual'],
        DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX,
    )

    print('\n--- UQ Results at deterministic design ---')
    print(f'  MTOM  mean : {meanW/G:.3f} kg   ({meanW:.2f} N)')
    print(f'  MTOM  std  : {stdW/G:.3f} kg')
    print(f'  95% PI     : [{p2_5:.3f}, {p97_5:.3f}] kg')
    print(f'  Det. MTOM  : {DET_W_N/G:.3f} kg  (reference)')
    print()
    print(sr.summary())

    print('\n' + '=' * 65)
    print('CONSTRAINT VIOLATION SUMMARY')
    print('=' * 65)
    print(f"  {'Constraint':<18} {'Limit':>8}  {'Violations':>10}  {'Rate':>8}")
    print('  ' + '-' * 50)
    print(f"  {'Cruise CL':<18} {CL_MAX:>8.3f}  {n_viol_cl:>10}  {pct_viol_cl:>7.2f} %")
    print(f"  {'Disk Loading':<18} {DL_MAX:>8.1f}  {n_viol_dl:>10}  {pct_viol_dl:>7.2f} %")
    print(f"  {'Blade Loading':<18} {BL_MAX:>8.4f}  {n_viol_bl:>10}  {pct_viol_bl:>7.2f} %")
    print('=' * 65)

    # ── Sensitivity ──────────────────────────────────────────────────────────
    corr_T   = float(np.corrcoef(T_arr,   W_kg)[0, 1])
    corr_ETA = float(np.corrcoef(ETA_arr, W_kg)[0, 1])
    print(f'\nPearson correlation with MTOM:')
    print(f'  T_HOVER   : {corr_T:+.3f}  (expected positive: more hover → heavier)')
    print(f'  ETA_HOVER : {corr_ETA:+.3f}  (expected negative: better efficiency → lighter)')
    print(f'  Ratio |corr_T / corr_ETA| = {abs(corr_T/corr_ETA):.2f}  '
          f'(relative influence of T_HOVER vs ETA_HOVER on MTOM)')

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir = Path(HERE) / 'uq_outputs'
    out_dir.mkdir(exist_ok=True)

    COLORS = {
        'mtom': '#4c72b0',
        'cl':   '#55a868',
        'dl':   '#dd8452',
        'bl':   '#c44e52',
    }

    # 1. MTOM histogram ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    _hist_with_kde(ax, W_kg, COLORS['mtom'])
    ax.axvline(np.mean(W_kg), color='red',    ls='--', lw=1.8,
               label=f'Mean {np.mean(W_kg):.3f} kg')
    ax.axvline(p2_5,          color='dimgray', ls=':',  lw=1.5,
               label=f'95% PI  [{p2_5:.3f}, {p97_5:.3f}] kg')
    ax.axvline(p97_5,         color='dimgray', ls=':',  lw=1.5)
    ax.axvline(DET_W_N / G,   color='purple',  ls='-.', lw=1.5,
               label=f'Deterministic {DET_W_N/G:.3f} kg')
    ax.set_xlim(*MTOM_LIM)
    ax.set_xlabel('MTOM (kg)');  ax.set_ylabel('Density')
    ax.set_title('MTOM distribution – joint (T_HOVER, η_hover) UQ')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_mtom_hist.png', dpi=180)
    plt.close(fig)
    print(f'\nSaved: {out_dir}/uq_mu_mtom_hist.png')

    # 2. Cruise CL ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    kde_cl = _hist_with_kde(ax, CL_arr, COLORS['cl'])
    _add_limit_line(ax, CL_MAX, f'Limit CL={CL_MAX:.2f}')
    _fill_violation(ax, kde_cl, CL_MAX, CL_LIM[1],
                    label=f'Violation {pct_viol_cl:.2f} %')
    ax.axvline(DET_CL, color='purple', ls='-.', lw=1.5,
               label=f'Deterministic CL={DET_CL:.3f}')
    ax.set_xlim(*CL_LIM)
    ax.set_xlabel('Cruise $C_L$');  ax.set_ylabel('Density')
    ax.set_title(f'Cruise $C_L$ distribution  (violations: {pct_viol_cl:.2f} %)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_cruiseCL_pdf.png', dpi=180)
    plt.close(fig)
    print(f'Saved: {out_dir}/uq_mu_cruiseCL_pdf.png')

    # 3. Disk Loading ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    kde_dl = _hist_with_kde(ax, DL_arr, COLORS['dl'])
    _add_limit_line(ax, DL_MAX, f'Limit {DL_MAX:.0f} N/m²')
    _fill_violation(ax, kde_dl, DL_MAX, DL_LIM[1],
                    label=f'Violation {pct_viol_dl:.2f} %')
    ax.set_xlim(*DL_LIM)
    ax.set_xlabel('Disk Loading (N/m²)');  ax.set_ylabel('Density')
    ax.set_title(f'Disk Loading distribution  (violations: {pct_viol_dl:.2f} %)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_diskloading_pdf.png', dpi=180)
    plt.close(fig)
    print(f'Saved: {out_dir}/uq_mu_diskloading_pdf.png')

    # 4. Blade Loading ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    kde_bl = _hist_with_kde(ax, BL_arr, COLORS['bl'])
    _add_limit_line(ax, BL_MAX, f'Limit {BL_MAX:.3f}')
    _fill_violation(ax, kde_bl, BL_MAX, BL_LIM[1],
                    label=f'Violation {pct_viol_bl:.2f} %')
    ax.set_xlim(*BL_LIM)
    ax.set_xlabel('Blade Loading');  ax.set_ylabel('Density')
    ax.set_title(f'Blade Loading distribution  (violations: {pct_viol_bl:.2f} %)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_bladeloading_pdf.png', dpi=180)
    plt.close(fig)
    print(f'Saved: {out_dir}/uq_mu_bladeloading_pdf.png')

    # 5. Scatter: T_HOVER vs MTOM (coloured by ETA_HOVER) + vice versa ────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    sc0 = ax.scatter(T_arr, W_kg, c=ETA_arr, cmap='RdYlGn',
                     s=10, alpha=0.55,
                     vmin=ETA_SCAT_LIM[0], vmax=ETA_SCAT_LIM[1])
    cb0 = plt.colorbar(sc0, ax=ax)
    cb0.set_label('η_hover [–]')
    ax.axhline(DET_W_N / G, color='purple', ls='-.', lw=1.2,
               label=f'Det. MTOM {DET_W_N/G:.3f} kg')
    ax.set_xlim(*T_HOV_SCAT_LIM)
    ax.set_xlabel('$T_{hover}$ (s)');  ax.set_ylabel('MTOM (kg)')
    ax.set_title('Effect of hover time on MTOM\n(colour = η_hover)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)

    ax = axes[1]
    sc1 = ax.scatter(ETA_arr, W_kg, c=T_arr, cmap='plasma',
                     s=10, alpha=0.55,
                     vmin=T_HOV_SCAT_LIM[0], vmax=T_HOV_SCAT_LIM[1])
    cb1 = plt.colorbar(sc1, ax=ax)
    cb1.set_label('$T_{hover}$ (s)')
    ax.axhline(DET_W_N / G, color='purple', ls='-.', lw=1.2,
               label=f'Det. MTOM {DET_W_N/G:.3f} kg')
    ax.axvline(eta_hover_deterministic_comparison, color='orange', ls='--', lw=1.2,
               label=f'Det. η_hover = {eta_hover_deterministic_comparison:.3f}')
    ax.set_xlim(*ETA_SCAT_LIM)
    ax.set_xlabel('η_hover [–]');  ax.set_ylabel('MTOM (kg)')
    ax.set_title('Effect of η_hover on MTOM\n(colour = T_HOVER)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.25)

    fig.suptitle('Joint (T_HOVER, η_hover) uncertainty  –  sensitivity scatter',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_scatter.png', dpi=180)
    plt.close(fig)
    print(f'Saved: {out_dir}/uq_mu_scatter.png')

    # 6. Input marginal distributions ─────────────────────────────────────────
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(10, 4))

    ax_t.hist(T_arr, bins=40, density=True, color='#4c72b0',
              alpha=0.55, edgecolor='k', linewidth=0.4)
    try:
        kde_t = gaussian_kde(T_arr)
        xs_t  = np.linspace(T_arr.min(), T_arr.max(), 400)
        ax_t.plot(xs_t, kde_t(xs_t), 'k-', lw=1.4)
    except Exception:
        pass
    ax_t.axvline(55.0, color='red', ls='--', lw=1.5, label='Mean 55 s')
    ax_t.set_xlabel('$T_{hover}$ (s)');  ax_t.set_ylabel('Density')
    ax_t.set_title('Hover time distribution\n(shifted lognormal)')
    ax_t.legend(fontsize=8);  ax_t.grid(alpha=0.25)

    ax_e.hist(ETA_arr, bins=40, density=True, color='#dd8452',
              alpha=0.55, edgecolor='k', linewidth=0.4)
    try:
        kde_e = gaussian_kde(ETA_arr)
        xs_e  = np.linspace(ETA_arr.min(), ETA_arr.max(), 400)
        ax_e.plot(xs_e, kde_e(xs_e), 'k-', lw=1.4)
    except Exception:
        pass
    ax_e.axvline(ETA_HOVER_MEAN, color='red', ls='--', lw=1.5,
                 label=f'Mean {ETA_HOVER_MEAN:.2f}')
    ax_e.axvline(eta_hover_deterministic_comparison, color='orange', ls=':', lw=1.5,
                 label=f'p2.5 = {eta_hover_deterministic_comparison:.3f}\n(det. comparison)')
    ax_e.set_xlabel('η_hover [–]');  ax_e.set_ylabel('Density')
    ax_e.set_title('Hover figure of merit distribution\n(truncated normal)')
    ax_e.legend(fontsize=8);  ax_e.grid(alpha=0.25)

    fig.suptitle('Uncertain input marginals (LHS samples)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'uq_mu_input_marginals.png', dpi=180)
    plt.close(fig)
    print(f'Saved: {out_dir}/uq_mu_input_marginals.png')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
