"""
sensitivity_oat.py  —  One-At-a-Time (OAT) sensitivity analysis
                        for the QBiT conceptual sizing model.

WHY WE RE-OPTIMISE FOR EACH PERTURBATION
-----------------------------------------
W_total is both the objective AND a state variable constrained by the
weight closure equation (weight_residual = 0).  Fixing the design point
and only calling run_model() leaves the weight loop open and infeasible —
W_total cannot respond to parameter changes.

The scientifically correct sensitivity for an optimisation-based sizing
model is the *optimal sensitivity*:

    S = d(W*_total) / d(param)

where W*_total is the optimal weight for each perturbed parameter set.
This requires running the full optimiser for every perturbation.

MONKEY-PATCHING WORKS HERE because build_qbit_model() is called INSIDE
run_optimisation(), AFTER the patch is applied.  The constants are read
fresh from the C module each time a new Problem is built.

RUNTIME
-------
18 params × 6 perturbations + 1 baseline = 109 optimisations.
Each SLSQP run takes ~0.1-0.5 s → total ~15-60 s on a laptop.

USAGE
-----
    python sensitivity_oat.py
    python sensitivity_oat.py --payload 3 --range 30 --nc 3
"""

from __future__ import annotations
import argparse
import time
import warnings
import numpy as np
import importlib
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import openmdao.api as om

# --- PATH FIX ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
om.config_reports = False


import openmdao.api as om
import qbit.constants as C
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import (
    G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
    J_BOUNDS, S_W_BOUNDS, DL_MAX, BL_MAX, CL_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# MISSION CONFIG  (overridable via CLI)
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--payload", type=float, default=3.0)
parser.add_argument("--range",   type=float, default=25.0,  help="km")
parser.add_argument("--nc",      type=int,   default=5)
args, _ = parser.parse_known_args()

PAYLOAD_KG = args.payload
RANGE_M    = args.range * 1_000.0
N_C        = args.nc

# Results directory named from CLI args: payload, n_c, range (km)
RANGE_KM = int(RANGE_M / 1_000.0)
PAYLOAD_STR = f"{PAYLOAD_KG:g}"
RESULTS_DIR = os.path.join(parent_dir, "sa", f"results_sa_{PAYLOAD_STR}_{N_C}_{RANGE_KM}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Matplotlib global style: white background, Times New Roman, readable sizes
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER TABLE
# Columns: (const_attr, nominal, label, type_tag, perturb_abs)
#   const_attr  : attribute name in qbit.constants
#   nominal     : nominal value (must match what is currently in constants.py)
#   label       : display name for plots/table
#   type_tag    : "input" | "param" | "model"
#   perturb_abs : True for negative-valued constants (K_WING_A) so the
#                 perturbation fraction applies to |nominal|, preserving sign
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = [
    ("RHO_AIR",         1.225,    "ρ_air",                "input", False),
    ("T_HOVER",         60.0,     "t_hover",              "input", False),
    ("BETA_QBIT",       0.18,     "β_QBiT (frame frac.)", "param", False),
    ("ETA_HOVER",       0.65,     "η_hover",              "param", False),
    ("CD0_WING",        0.01,     "CD0_wing",             "param", False),
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", False),
    ("AR_FIXED",        8.0,      "AR",                   "input", False),
    ("SIGMA",           0.13,     "σ (solidity)",         "param", False),
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", False),
    ("KAPPA_MAX",       1.15,     "κ_max",                "param", False),
    ("BATTERY_DENSITY", 158.0,    "ρ_bat (Wh/kg)",        "input", False),
    ("BATTERY_EFF",     0.85,     "η_bat",                "param", False),
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", False),
    ("K_ESC",           3.594e-4, "k_ESC",                "model", False),
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", False),
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", False),
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", True),
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", False),
]

PERTURB_FRACS = np.array([-0.20, -0.10, -0.05, +0.05, +0.10, +0.20])

TYPE_COLORS = {
    "input": "#388bfd",
    "param": "#d29922",
    "model": "#a371f7",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _restore_all() -> None:
    for attr, nominal, *_ in PARAMS:
        setattr(C, attr, nominal)


def _apply_perturbation(attr: str, nominal: float,
                         frac: float, perturb_abs: bool) -> float:
    if perturb_abs:
        perturbed = nominal + frac * abs(nominal)
    else:
        perturbed = nominal * (1.0 + frac)
    setattr(C, attr, perturbed)
    return perturbed


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE OPTIMISATION RUN
# The key insight: build_qbit_model() is called HERE, after the patch,
# so it reads the current (perturbed) C.* values at construction time.
# ─────────────────────────────────────────────────────────────────────────────
def optimise(payload_kg: float, range_m: float, n_c: int) -> tuple[float, bool]:
    # Ensure any modules that imported constants via
    # "from qbit.constants import ..." are reloaded so they pick up
    # the current values in the qbit.constants module.
    # Do NOT reload qbit.constants here — it contains the patched values.
    # Reload other qbit submodules so they re-import names from the
    # already-modified qbit.constants module.
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass

    # Re-import the model builder so we get the reloaded module's symbols
    try:
        from qbit.models.qbit_model import build_qbit_model as _build_qbit_model
    except Exception:
        _build_qbit_model = build_qbit_model

    # optional debug: set env SENS_OAT_DEBUG=1 to print reloaded constant values
    if os.getenv("SENS_OAT_DEBUG") == "1":
        try:
            import qbit.components.hover_power as _hp
            hp_rho = getattr(_hp, 'RHO_AIR', '<missing>')
        except Exception:
            hp_rho = '<no-module>'
        print(f"[DEBUG] qbit.constants.RHO_AIR={getattr(C, 'RHO_AIR', None)}  ")
        print(f"[DEBUG] hover_power.RHO_AIR={hp_rho}")

    prob = om.Problem(reports=None)
    prob.model = _build_qbit_model(payload_kg, range_m, n_c)  # now reads C.* at import time

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"]       = 1e-9
    prob.driver.options["maxiter"]   = 2000

    prob.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var("V_inf",   lower=V_INF_BOUNDS[0],   upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r",       lower=R_BOUNDS[0],       upper=R_BOUNDS[1])
    prob.model.add_design_var("J",       lower=J_BOUNDS[0],       upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w",     lower=S_W_BOUNDS[0],     upper=S_W_BOUNDS[1])

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading",    upper=DL_MAX)
    prob.model.add_constraint("blade_loading",   upper=BL_MAX)
    prob.model.add_constraint("cruise_CL",       upper=CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    prob.setup()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()

    W_opt = float(prob.get_val("W_total")[0])
    try:
        converged = bool(prob.driver.result.success)
    except AttributeError:
        converged = True
    return W_opt, converged


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def baseline() -> float:
    _restore_all()
    W, conv = optimise(PAYLOAD_KG, RANGE_M, N_C)
    if not conv:
        print("  [WARN] Baseline optimisation did not report clean convergence.")
    return W


# ─────────────────────────────────────────────────────────────────────────────
# OAT SWEEP
# ─────────────────────────────────────────────────────────────────────────────
def oat_sweep(W_base: float) -> list[dict]:
    results   = []
    n_total   = len(PARAMS) * len(PERTURB_FRACS)
    run_count = 0
    t0        = time.time()

    for attr, nominal, label, type_tag, perturb_abs in PARAMS:
        S_list, dW_list, W_list = [], [], []
        any_fail = False

        for frac in PERTURB_FRACS:
            run_count += 1
            elapsed = time.time() - t0
            eta = (elapsed / run_count) * (n_total - run_count) if run_count > 1 else 0.0
            print(f"  [{run_count:>3}/{n_total}] {label:28s} "
                  f"Δ={frac:+.0%}  ETA {eta:.0f}s …", end="\r", flush=True)

            _restore_all()
            p_val = _apply_perturbation(attr, nominal, frac, perturb_abs)

            W_opt, conv = optimise(PAYLOAD_KG, RANGE_M, N_C)
            if not conv:
                any_fail = True

            dW = 100.0 * (W_opt - W_base) / W_base
            actual_frac = (p_val - nominal) / abs(nominal) if nominal != 0 else frac
            S  = dW / (actual_frac * 100.0) if actual_frac != 0 else 0.0

            S_list.append(S)
            dW_list.append(dW)
            W_list.append(W_opt)

        _restore_all()

        S_mean = float(np.mean(np.abs(S_list)))
        # summary line (overwrites the ETA line)
        flag = " ◀◀" if S_mean >= 0.5 else (" ◀" if S_mean >= 0.1 else "")
        warn = " !" if any_fail else ""
        print(f"  ✓  {label:30s}  |S̄|={S_mean:.4f}  "
              f"ΔW@±10%=[{dW_list[1]:+.2f}%, {dW_list[4]:+.2f}%]{flag}{warn}    ")

        results.append({
            "attr":    attr,
            "label":   label,
            "type":    type_tag,
            "nominal": nominal,
            "S_mean":  S_mean,
            "S_vals":  S_list,
            "dW_pcts": dW_list,
            "W_opts":  W_list,
            "failed":  any_fail,
        })

    results.sort(key=lambda r: r["S_mean"], reverse=True)
    print(f"\n  Sweep complete — {time.time()-t0:.1f} s total\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results: list[dict], W_base: float) -> None:
    SEP = "=" * 84
    print(f"\n{SEP}")
    print(f"  OAT Sensitivity of Optimal MTOM  —  baseline = {W_base/G:.4f} kg")
    print(f"  Mission: payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print(SEP)
    print(f"  {'Rk':>2}  {'Parameter':30s}  {'Type':6s}  "
          f"{'|S̄|':>7}  {'ΔW@−10%':>9}  {'ΔW@+10%':>9}  {'ΔW@+20%':>9}")
    print("  " + "─" * 78)
    for i, r in enumerate(results, 1):
        dw  = r["dW_pcts"]
        flg = " ◀◀" if r["S_mean"] >= 0.5 else (" ◀" if r["S_mean"] >= 0.1 else "")
        print(f"  {i:>2}  {r['label']:30s}  {r['type']:6s}  "
              f"{r['S_mean']:>7.4f}  "
              f"{dw[1]:>+8.2f}%  "
              f"{dw[4]:>+8.2f}%  "
              f"{dw[5]:>+8.2f}%{flg}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _styled_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    return fig, ax


def plot_ranked_bar(results, W_base):
    from matplotlib.patches import Patch
    labels = [r["label"] for r in results]
    vals   = [r["S_mean"] for r in results]
    colors = [TYPE_COLORS.get(r["type"], "#2f4f4f") for r in results]

    fig, ax = _styled_fig(h=max(5, len(labels) * 0.42))
    y    = np.arange(len(labels))
    xmax = max(vals) if max(vals) > 0 else 1.0

    ax.barh(y, vals, color=colors, height=0.62, edgecolor="none")
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yi, f"{v:.4f}",
                va="center", fontsize=10, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color="black")
    ax.set_xlabel("|S̄| = mean|%ΔW*_total / %Δparam|  (±5 %, ±10 %, ±20 %)",
                  color="black", fontsize=12)
    ax.set_title(
        f"OAT Parameter Ranking  —  Optimal MTOM = {W_base/G:.3f} kg  "
        f"(payload={PAYLOAD_KG} kg, R={RANGE_M/1e3:.0f} km)",
        color="black", fontsize=14, pad=10)
    ax.legend(
        handles=[Patch(fc=TYPE_COLORS[t], label=lb) for t, lb in [
            ("input", "Input / mission"), ("param", "Parametric"),
            ("model", "Model-form")]],
        facecolor="white", edgecolor="black", fontsize=11, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.22)
    ax.axvline(0.05, color="black", lw=0.9, ls=(0, (5, 5)))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sensitivity_ranked_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {out}")
    plt.close()


def plot_tornado(results):
    top    = results[:min(12, len(results))]
    labels = [r["label"] for r in top]
    lo     = [r["dW_pcts"][1] for r in top]   # Δ = −10 %
    hi     = [r["dW_pcts"][4] for r in top]   # Δ = +10 %
    colors = [TYPE_COLORS.get(r["type"], "#2f4f4f") for r in top]

    fig, ax = _styled_fig(h=max(4, len(top) * 0.45))
    for i, (l, h, col) in enumerate(zip(lo, hi, colors)):
        ax.barh(i, abs(h - l), left=min(l, h), color=col,
                height=0.55, alpha=0.85, edgecolor="none")
        sp = max(abs(l), abs(h), 0.05)
        ax.text(l - sp * 0.05, i, f"{l:+.2f}%",
            va="center", ha="right", fontsize=10, color="black")
        ax.text(h + sp * 0.05, i, f"{h:+.2f}%",
            va="center", ha="left",  fontsize=10, color="black")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=11, color="black")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("ΔW*_total [%]  (parameter perturbed ±10 %)",
                  color="black", fontsize=12)
    ax.set_title("Tornado diagram  —  optimal MTOM sensitivity (±10 %)",
                 color="black", fontsize=14, pad=10)
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sensitivity_tornado_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {out}")
    plt.close()


def plot_sweep_lines(results, W_base, top_n=6):
    top     = results[:top_n]
    fx      = np.array(PERTURB_FRACS) * 100.0
    base_kg = W_base / G
    palette = ["#388bfd", "#d29922", "#a371f7", "#3fb950", "#f78166", "#79c0ff"]

    fig, ax = _styled_fig(w=9, h=5)
    for r, col in zip(top, palette):
        ax.plot(fx, [w / G for w in r["W_opts"]],
                marker="o", ms=4.5, lw=1.8, color=col, label=r["label"])

    ax.axhline(base_kg, color="black", lw=0.8, ls=(0, (5, 5)),
               label=f"Baseline ({base_kg:.3f} kg)")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Parameter perturbation [%]", color="black", fontsize=12)
    ax.set_ylabel("Optimal W_total [kg]",        color="black", fontsize=12)
    ax.set_title(f"Optimal MTOM vs parameter perturbation — top {top_n}",
                 color="black", fontsize=14, pad=10)
    ax.legend(facecolor="white", edgecolor="black",
              fontsize=11)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sensitivity_sweeplines_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {out}")
    plt.close()


def plot_heatmap(results):
    """Signed elasticity heatmap — shows nonlinearity and sign."""
    fracs_lbl = [f"{f:+.0%}" for f in PERTURB_FRACS]
    labels    = [r["label"] for r in results]
    data      = np.array([r["S_vals"] for r in results])

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.38)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vmax = np.nanmax(np.abs(data)) or 1.0
    im   = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Elasticity  S = %ΔW* / %Δparam",
                   color="black", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=9)

    ax.set_xticks(range(len(fracs_lbl)))
    ax.set_xticklabels(fracs_lbl, color="black", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="black", fontsize=10)
    ax.set_xlabel("Parameter perturbation", color="black", fontsize=12)
    ax.set_title("Elasticity heatmap  —  S = %ΔW*_total / %Δparam",
                 color="black", fontsize=14, pad=10)

    for i in range(len(labels)):
        for j in range(len(fracs_lbl)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(data[i, j]) > vmax * 0.5 else "#8b949e")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"sensitivity_heatmap_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {out}")
    plt.close()


def write_ranked_csv(results):
    import csv
    outcsv = os.path.join(RESULTS_DIR, f"sensitivity_ranked_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.csv")
    with open(outcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attr", "label", "type", "nominal", "S_mean", "dW_minus10_pct", "dW_plus10_pct", "dW_plus20_pct", "failed"])
        for i, r in enumerate(results, 1):
            dw = r["dW_pcts"]
            w.writerow([i, r["attr"], r["label"], r["type"], r["nominal"], f"{r['S_mean']:.6f}", f"{dw[1]:.6f}", f"{dw[4]:.6f}", f"{dw[5]:.6f}", r["failed"]])
    print(f"  → {outcsv}")


# ─────────────────────────────────────────────────────────────────────────────
# UMDO GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────
def umdo_guidance(results: list[dict]) -> None:
    print("\n── UMDO parameter selection guidance ───────────────────────────")
    print("  Metric: |S̄| = mean|%ΔW*_total / %Δparam|  (averaged over ±5,10,20%)\n")

    print("  Top-3 → candidates for uncertainty propagation:")
    for i, r in enumerate(results[:3], 1):
        band = r["dW_pcts"][4] - r["dW_pcts"][1]
        print(f"    {i}. {r['label']:30s}  |S̄|={r['S_mean']:.4f}  "
              f"ΔW band @±10% = {band:.2f}%  [{r['type']}]")

    freeze = [r for r in results if r["S_mean"] < 0.02]
    if freeze:
        print(f"\n  Safe to freeze ({len(freeze)} params with |S̄| < 0.02):")
        for r in freeze:
            print(f"    – {r['label']:30s}  |S̄|={r['S_mean']:.4f}")

    mid = [r for r in results if 0.02 <= r["S_mean"] < 0.10]
    if mid:
        print(f"\n  Marginal (0.02 ≤ |S̄| < 0.10):")
        for r in mid:
            print(f"    ~ {r['label']:30s}  |S̄|={r['S_mean']:.4f}")

    print(f"\n  → For 1-param UMDO: {results[0]['label']}")
    if len(results) > 1:
        print(f"  → For 2-param UMDO: {results[0]['label']}  +  {results[1]['label']}")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── QBiT OAT Sensitivity Analysis ───────────────────────────────")
    print(f"   Mission:  payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print(f"   {len(PARAMS)} parameters × {len(PERTURB_FRACS)} perturbations "
          f"+ 1 baseline = {len(PARAMS)*len(PERTURB_FRACS)+1} optimisations")
    print("────────────────────────────────────────────────────────────────\n")

    print("Baseline …")
    t0     = time.time()
    W_base = baseline()
    print(f"  W*_total = {W_base/G:.4f} kg  ({time.time()-t0:.1f} s)\n")

    print("OAT sweep …\n")
    results = oat_sweep(W_base)

    print_table(results, W_base)

    print("\nGenerating plots …")
    plot_ranked_bar(results, W_base)
    plot_tornado(results)
    plot_sweep_lines(results, W_base)
    plot_heatmap(results)
    write_ranked_csv(results)

    umdo_guidance(results)