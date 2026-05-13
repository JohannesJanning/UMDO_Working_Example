"""
sensitivity_totals.py
─────────────────────
Rigorous local sensitivity analysis using OpenMDAO's analytic total
derivatives (prob.compute_totals) at the deterministic optimum.

WHAT THIS COMPUTES
──────────────────
After running the deterministic optimisation to convergence, we compute:

    dW*_total / d(param_i)   for every uncertain parameter i

These are *total* derivatives — they propagate through the full coupled
model including the implicit weight-closure loop.  They are computed by
OpenMDAO's adjoint/forward linear solver (analytic, not finite-difference)
so they are exact to machine precision.

The result is a *local* sensitivity at the deterministic optimum x_det.
It answers: "at this specific design point, how does W_total change if
param_i shifts by a small amount?"

CROSS-VALIDATION AGAINST OAT
──────────────────────────────
The analytic gradient dW*/dp can be compared to the OAT finite-difference
result:

    S_OAT = (W*(p+Δp) - W*(p-Δp)) / (2Δp)     [central difference]

If both agree to within ~1%, the model partials are correct and the OAT
results are trustworthy.  Agreement gives you high confidence before
committing to UQ propagation.

WHY NOT JUST USE OAT?
──────────────────────
OAT requires 6 optimiser runs per parameter (±5%, ±10%, ±20%) → 109 total.
compute_totals requires ONE model evaluation + ONE linear solve → ~0.01 s.
The analytic result is also exact, while OAT finite-differences have
truncation error from the finite step size.

LIMITATION
──────────────────────────────
compute_totals gives LOCAL sensitivities (linearised at x_det).  For
nonlinear models the sensitivity changes across the parameter space.
That is why the OAT sweep (which samples ±20%) is complementary — it
reveals nonlinearity.  Use BOTH together.

REQUIREMENT
──────────────────────────────
The uncertain parameters must be OpenMDAO *inputs* somewhere in the model
(even if currently hardcoded from constants).  The promoted input name is
what you pass to compute_totals(wrt=[...]).

Run  python sensitivity_totals.py --list-inputs  to print all input names.

USAGE
──────────────────────────────
    python sensitivity_totals.py
    python sensitivity_totals.py --payload 3 --range 30 --nc 3
    python sensitivity_totals.py --list-inputs
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
import importlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openmdao.api as om

# ── path fix ────────────────────────────────────────────────────────────────
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
om.config_reports = False

import qbit.constants as C
from qbit.constants import (
    G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
    J_BOUNDS, S_W_BOUNDS, 
)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--payload",     type=float, default=3.0)
parser.add_argument("--range",       type=float, default=15.0, help="km one-way")
parser.add_argument("--nc",          type=int,   default=2)
parser.add_argument("--list-inputs", action="store_true",
                    help="Print all promoted model inputs and exit")
parser.add_argument("--fd-step",     type=float, default=0.01,
                    help="Finite-difference step fraction for cross-validation (default 1%%)")
args, _ = parser.parse_known_args()

PAYLOAD_KG = args.payload
RANGE_M    = args.range * 1_000.0
N_C        = args.nc
FD_FRAC    = args.fd_step          # relative step for FD cross-check

RANGE_KM   = int(RANGE_M / 1_000)
RESULTS_DIR = os.path.join(
    parent_dir, "sa",
    f"results_totals_{PAYLOAD_KG:g}_{N_C}_{RANGE_KM}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 1.8, "lines.markersize": 6,
})

# ────────────────────────────────────────────────────────────────────────────
# UNCERTAIN PARAMETER TABLE
# ────────────────────────────────────────────────────────────────────────────
# Each entry: (const_attr, nominal, label, type_tag, om_input_name)
#
# om_input_name: the PROMOTED name of the OpenMDAO input that carries this
#   constant inside the model.  Find it by running --list-inputs.
#   Set to None if the constant is not yet wired as an OM input — those
#   parameters will be handled via finite-difference re-optimisation
#   (fallback) and flagged clearly in the output.
#
# NOTE: The names below are placeholders.  Run --list-inputs first and
#       replace them with the actual promoted names from your model.
# ────────────────────────────────────────────────────────────────────────────
PARAMS = [
    # attr              nominal    label                    type     om_name
    ("RHO_AIR",         1.225,    "ρ_air",                "input", None),
    ("T_HOVER",         60.0,     "t_hover",              "input", None),
    ("BETA_QBIT",       0.18,     "β_QBiT (frame frac.)", "param", None),
    ("ETA_HOVER",       0.65,     "η_hover",              "param", None),
    ("CD0_WING",        0.01,     "CD0_wing",             "param", None),
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", None),
    ("AR_FIXED",        8.0,      "AR",                   "input", None),
    ("SIGMA",           0.13,     "σ (solidity)",         "param", None),
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", None),
    ("KAPPA_MAX",       1.15,     "κ_max",                "param", None),
    ("BATTERY_DENSITY", 158.0,    "ρ_bat (Wh/kg)",        "input", None),
    ("BATTERY_EFF",     0.85,     "η_bat",                "param", None),
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", None),
    ("K_ESC",           3.594e-4, "k_ESC",                "model", None),
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", None),
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", None),
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", None),
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", None),
]

PARAMS += [
    ("DL_MAX", C.DL_MAX, "DL_MAX (disk loading)", "param", None),
    ("BL_MAX", C.BL_MAX, "BL_MAX (blade loading)", "param", None),
    ("CL_MAX", C.CL_MAX, "CL_MAX (cruise CL)", "param", None),
]

TYPE_COLORS = {
    "input": "#1f77b4",
    "param": "#d29922",
    "model": "#7b2d8b",
}


# ────────────────────────────────────────────────────────────────────────────
# RELOAD HELPER  (same mechanism as sensitivity_oat.py)
# ────────────────────────────────────────────────────────────────────────────
def _reload_qbit_modules() -> None:
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass


def _restore_constants() -> None:
    for attr, nominal, *_ in PARAMS:
        setattr(C, attr, nominal)
    _reload_qbit_modules()


# ────────────────────────────────────────────────────────────────────────────
# BUILD AND RUN THE DETERMINISTIC OPTIMISATION
# ────────────────────────────────────────────────────────────────────────────
def build_and_optimise() -> om.Problem:
    """
    Run the deterministic optimisation and return the converged Problem
    object.  The Problem is kept alive so we can call compute_totals on it.
    """
    _restore_constants()
    _reload_qbit_modules()
    from qbit.models.qbit_model import build_qbit_model

    prob = om.Problem(reports=None)
    prob.model = build_qbit_model(PAYLOAD_KG, RANGE_M, N_C)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"]       = 1e-9
    prob.driver.options["maxiter"]   = 2000

    prob.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0],
                              upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var("V_inf",   lower=V_INF_BOUNDS[0],
                              upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r",       lower=R_BOUNDS[0], upper=R_BOUNDS[1])
    prob.model.add_design_var("J",       lower=J_BOUNDS[0], upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w",     lower=S_W_BOUNDS[0],
                              upper=S_W_BOUNDS[1])

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading",    upper=C.DL_MAX)
    prob.model.add_constraint("blade_loading",   upper=C.BL_MAX)
    prob.model.add_constraint("cruise_CL",       upper=C.CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    prob.setup(force_alloc_complex=True)   # needed for complex-step validation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()

    return prob


# ────────────────────────────────────────────────────────────────────────────
# LIST ALL PROMOTED INPUTS  (--list-inputs mode)
# ────────────────────────────────────────────────────────────────────────────
def list_inputs(prob: om.Problem) -> None:
    print("\n── All promoted scalar inputs in the model ──────────────────────")
    print(f"  {'promoted name':50s}  {'value':>14s}  {'units'}")
    print("  " + "─" * 75)
    data = prob.model.list_inputs(val=True, prom_name=True,
                                  units=True, out_stream=None)
    seen = set()
    for _, meta in sorted(data, key=lambda x: x[1].get("prom_name", "")):
        pname = meta.get("prom_name", "?")
        if pname in seen:
            continue
        seen.add(pname)
        val = np.atleast_1d(meta.get("val", [np.nan]))
        units = meta.get("units", "")
        if val.size == 1:
            print(f"  {pname:50s}  {float(val[0]):>14.6g}  {units}")
    print()
    print("  Copy the relevant promoted names into the om_name column of PARAMS.")
    print("─────────────────────────────────────────────────────────────────────\n")


# ────────────────────────────────────────────────────────────────────────────
# ANALYTIC TOTAL DERIVATIVES  via compute_totals
# ────────────────────────────────────────────────────────────────────────────
def compute_analytic_sensitivities(prob: om.Problem,
                                   W_opt: float) -> list[dict]:
    """
    Call prob.compute_totals(of=['W_total'], wrt=[om_name]) for every
    parameter that has a known om_name.

    Returns list of result dicts with keys:
        label, type, nominal, om_name,
        dW_dP        [N / unit_of_param]
        S_analytic   [normalised: (dW/W) / (dp/p), dimensionless]
    """
    results = []
    wired   = [(attr, nom, lbl, typ, omn)
               for attr, nom, lbl, typ, omn in PARAMS if omn is not None]

    if not wired:
        print("  [WARN] No om_names defined in PARAMS.")
        print("  Run --list-inputs to find the correct promoted names,")
        print("  then fill in the om_name column in PARAMS.")
        return results

    print(f"  Computing analytic dW*/dp for {len(wired)} wired parameters …\n")

    for attr, nominal, label, type_tag, om_name in wired:
        try:
            totals = prob.compute_totals(
                of=["W_total"],
                wrt=[om_name],
                return_format="flat_dict",
            )
            dW_dP = float(totals[("W_total", om_name)].ravel()[0])
        except Exception as e:
            print(f"  [FAIL] {label:30s}  compute_totals error: {e}")
            continue

        # Normalised elasticity: (ΔW/W) / (Δp/p)  at nominal
        S = (dW_dP * abs(nominal) / W_opt) if nominal != 0 else 0.0

        results.append({
            "attr":       attr,
            "label":      label,
            "type":       type_tag,
            "nominal":    nominal,
            "om_name":    om_name,
            "dW_dP":      dW_dP,       # dimensional: N per unit_of_param
            "S_analytic": S,           # dimensionless elasticity
        })
        print(f"  ✓  {label:30s}  dW/dp = {dW_dP:+.4e}  S = {S:+.4f}")

    results.sort(key=lambda r: abs(r["S_analytic"]), reverse=True)
    return results


# ────────────────────────────────────────────────────────────────────────────
# FD CROSS-VALIDATION  (re-optimise at ±FD_FRAC perturbation)
# ────────────────────────────────────────────────────────────────────────────
def fd_crossvalidation(analytic_results: list[dict],
                       W_opt: float) -> list[dict]:
    """
    For each parameter with an analytic result, compute a central-difference
    finite-difference gradient by re-running the optimiser at ±FD_FRAC.
    Compare to the analytic value.

    This is the cross-validation step: agreement within ~1 % means the
    model partials are correct and the analytic sensitivities are trusted.
    """
    from qbit.models.qbit_model import build_qbit_model

    def _run_opt(attr, value):
        _restore_constants()
        setattr(C, attr, value)
        _reload_qbit_modules()
        from qbit.models.qbit_model import build_qbit_model as _bld

        p = om.Problem(reports=None)
        p.model = _bld(PAYLOAD_KG, RANGE_M, N_C)
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options["optimizer"] = "SLSQP"
        p.driver.options["tol"]       = 1e-9
        p.driver.options["maxiter"]   = 2000

        p.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0],
                               upper=W_TOTAL_BOUNDS[1])
        p.model.add_design_var("V_inf",   lower=V_INF_BOUNDS[0],
                               upper=V_INF_BOUNDS[1])
        p.model.add_design_var("r",       lower=R_BOUNDS[0], upper=R_BOUNDS[1])
        p.model.add_design_var("J",       lower=J_BOUNDS[0], upper=J_BOUNDS[1])
        p.model.add_design_var("S_w",     lower=S_W_BOUNDS[0],
                               upper=S_W_BOUNDS[1])
        p.model.add_objective("W_total")
        p.model.add_constraint("weight_residual", equals=0.0)
        p.model.add_constraint("disk_loading",    upper=C.DL_MAX)
        p.model.add_constraint("blade_loading",   upper=C.BL_MAX)
        p.model.add_constraint("cruise_CL",       upper=C.CL_MAX)
        p.model.set_input_defaults("W_total", val=6.0 * G, units="N")
        p.setup()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.run_driver()
        return float(p.get_val("W_total")[0])

    print(f"\n  FD cross-validation  (central difference, step = ±{FD_FRAC:.1%}) …\n")
    validated = []

    for r in analytic_results:
        attr    = r["attr"]
        nominal = r["nominal"]
        p_plus  = nominal * (1.0 + FD_FRAC) if nominal >= 0 \
                  else nominal * (1.0 - FD_FRAC)
        p_minus = nominal * (1.0 - FD_FRAC) if nominal >= 0 \
                  else nominal * (1.0 + FD_FRAC)
        dp      = p_plus - p_minus           # actual Δp for CD formula

        W_plus  = _run_opt(attr, p_plus)
        W_minus = _run_opt(attr, p_minus)
        _restore_constants()

        dW_dP_fd = (W_plus - W_minus) / dp  # central difference
        S_fd     = (dW_dP_fd * abs(nominal) / W_opt) if nominal != 0 else 0.0

        S_analytic = r["S_analytic"]
        rel_err    = abs(S_analytic - S_fd) / max(abs(S_fd), 1e-12)

        flag = "✓" if rel_err < 0.02 else ("~" if rel_err < 0.10 else "✗")
        print(f"  {flag}  {r['label']:30s}  "
              f"S_analytic={S_analytic:+.4f}  "
              f"S_FD={S_fd:+.4f}  "
              f"rel_err={rel_err:.2%}")

        rv = dict(r)
        rv["S_fd"]    = S_fd
        rv["rel_err"] = rel_err
        rv["agreed"]  = rel_err < 0.02
        validated.append(rv)

    _restore_constants()
    return validated


# ────────────────────────────────────────────────────────────────────────────
# FALLBACK: FD-ONLY FOR UNWIRED PARAMETERS
# ────────────────────────────────────────────────────────────────────────────
def fd_only_sensitivity(W_opt: float) -> list[dict]:
    """
    For parameters without om_name (not wired as OM inputs), compute
    sensitivity purely via re-optimisation FD.  Same as OAT but with a
    single central-difference step at FD_FRAC.
    """
    unwired = [(attr, nom, lbl, typ)
               for attr, nom, lbl, typ, omn in PARAMS if omn is None]
    if not unwired:
        return []

    print(f"\n  FD-only sensitivity for {len(unwired)} unwired parameters "
          f"(step ±{FD_FRAC:.1%}) …\n")
    results = []

    for attr, nominal, label, type_tag in unwired:
        p_plus  = nominal * (1.0 + FD_FRAC)
        p_minus = nominal * (1.0 - FD_FRAC)
        dp      = p_plus - p_minus

        # plus
        _restore_constants(); setattr(C, attr, p_plus); _reload_qbit_modules()
        from qbit.models.qbit_model import build_qbit_model as _bld

        def _run():
            p = om.Problem(reports=None); p.model = _bld(PAYLOAD_KG, RANGE_M, N_C)
            p.driver = om.ScipyOptimizeDriver()
            p.driver.options["optimizer"] = "SLSQP"
            p.driver.options["tol"] = 1e-9; p.driver.options["maxiter"] = 2000
            p.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
            p.model.add_design_var("V_inf",   lower=V_INF_BOUNDS[0],   upper=V_INF_BOUNDS[1])
            p.model.add_design_var("r",       lower=R_BOUNDS[0],        upper=R_BOUNDS[1])
            p.model.add_design_var("J",       lower=J_BOUNDS[0],        upper=J_BOUNDS[1])
            p.model.add_design_var("S_w",     lower=S_W_BOUNDS[0],      upper=S_W_BOUNDS[1])
            p.model.add_objective("W_total")
            p.model.add_constraint("weight_residual", equals=0.0)
            p.model.add_constraint("disk_loading",    upper=C.DL_MAX)
            p.model.add_constraint("blade_loading",   upper=C.BL_MAX)
            p.model.add_constraint("cruise_CL",       upper=C.CL_MAX)
            p.model.set_input_defaults("W_total", val=6.0 * G, units="N")
            p.setup()
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); p.run_driver()
            return float(p.get_val("W_total")[0])

        W_plus = _run()
        _restore_constants(); setattr(C, attr, p_minus); _reload_qbit_modules()
        from qbit.models.qbit_model import build_qbit_model as _bld  # re-import after reload
        W_minus = _run()
        _restore_constants()

        dW_dP = (W_plus - W_minus) / dp
        S_fd  = (dW_dP * abs(nominal) / W_opt) if nominal != 0 else 0.0
        print(f"  ○  {label:30s}  S_FD={S_fd:+.4f}  (unwired — FD only)")
        results.append({
            "attr": attr, "label": label, "type": type_tag,
            "nominal": nominal, "om_name": None,
            "S_analytic": None, "S_fd": S_fd, "rel_err": None, "agreed": None,
        })

    return results


# ────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ────────────────────────────────────────────────────────────────────────────
def print_table(all_results: list[dict], W_opt: float) -> None:
    # Sort by |S| — use S_fd if S_analytic unavailable
    def _s(r):
        v = r.get("S_analytic") or r.get("S_fd") or 0.0
        return abs(v)
    all_results.sort(key=_s, reverse=True)

    SEP = "=" * 90
    print(f"\n{SEP}")
    print(f"  Sensitivity of Optimal MTOM  —  W*_total = {W_opt/G:.4f} kg")
    print(f"  Mission: payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print(f"  Method: analytic total derivatives (compute_totals) + FD cross-check")
    print(SEP)
    print(f"  {'Rk':>2}  {'Parameter':30s}  {'Type':6s}  "
          f"{'S_analytic':>12}  {'S_FD':>10}  {'rel_err':>9}  {'OK?':>5}")
    print("  " + "─" * 84)
    for i, r in enumerate(all_results, 1):
        Sa  = f"{r['S_analytic']:+.4f}" if r["S_analytic"] is not None else "  n/a  "
        Sfd = f"{r['S_fd']:+.4f}"       if r.get("S_fd")   is not None else "  n/a  "
        er  = f"{r['rel_err']:.2%}"     if r.get("rel_err") is not None else "  n/a  "
        ok  = ("✓" if r["agreed"] else ("~" if r["agreed"] is False and
               abs(r.get("rel_err") or 1) < 0.10 else "✗")) \
              if r["agreed"] is not None else "○"
        print(f"  {i:>2}  {r['label']:30s}  {r['type']:6s}  "
              f"{Sa:>12}  {Sfd:>10}  {er:>9}  {ok:>5}")
    print(SEP)


# ────────────────────────────────────────────────────────────────────────────
# PLOT
# ────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_results: list[dict], W_opt: float) -> None:
    """
    Side-by-side bar chart: S_analytic vs S_FD for each parameter.
    Visual agreement validates the model.
    """
    wired = [r for r in all_results if r.get("S_analytic") is not None]
    if not wired:
        print("  [SKIP] No analytic results to plot.")
        return

    labels     = [r["label"] for r in wired]
    S_an       = [abs(r["S_analytic"]) for r in wired]
    S_fd       = [abs(r["S_fd"])       for r in wired]
    agreed     = [r["agreed"]          for r in wired]
    colors_an  = [TYPE_COLORS.get(r["type"], "gray") for r in wired]

    y     = np.arange(len(labels))
    h     = 0.35
    xmax  = max(max(S_an), max(S_fd)) if S_an else 1.0

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    bars_an = ax.barh(y + h/2, S_an, height=h, color=colors_an,
                      label="Analytic (compute_totals)", alpha=0.9)
    bars_fd = ax.barh(y - h/2, S_fd, height=h, color=colors_an,
                      label=f"FD re-optimise (±{FD_FRAC:.0%})",
                      alpha=0.45, hatch="//")

    # Annotate disagreements
    for yi, sa, sf, ok in zip(y, S_an, S_fd, agreed):
        if not ok:
            ax.text(max(sa, sf) + xmax * 0.02, yi, "mismatch",
                    va="center", fontsize=8, color="red")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("|S| = |(dW*/dp)·(p_nom/W*)| — normalised elasticity",
                  fontsize=12)
    ax.set_title(
        f"Analytic vs FD Sensitivity  —  W*_total = {W_opt/G:.3f} kg\n"
        f"payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}",
        fontsize=14)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.25)
    ax.axvline(0.05, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    fname = os.path.join(
        RESULTS_DIR,
        f"totals_vs_fd_{PAYLOAD_KG:g}_{N_C}_{RANGE_KM}km.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\n  → {fname}")
    plt.close()


def plot_ranking(all_results: list[dict], W_opt: float) -> None:
    """
    Final ranked bar chart using analytic S where available, FD otherwise.
    This is the definitive sensitivity ranking for UMDO parameter selection.
    """
    from matplotlib.patches import Patch

    def _best_s(r):
        return r["S_analytic"] if r.get("S_analytic") is not None \
               else r.get("S_fd", 0.0)

    data   = sorted(all_results, key=lambda r: abs(_best_s(r)), reverse=True)
    labels = [r["label"] for r in data]
    vals   = [abs(_best_s(r)) for r in data]
    colors = [TYPE_COLORS.get(r["type"], "gray") for r in data]
    xmax   = max(vals) if vals else 1.0

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.42)))
    ax.barh(np.arange(len(labels)), vals, color=colors, height=0.62,
            edgecolor="none")
    for yi, v in enumerate(vals):
        ax.text(v + xmax * 0.015, yi, f"{v:.4f}", va="center",
                fontsize=10, color="black")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("|S| = normalised elasticity  dW*/dp · p_nom/W*", fontsize=12)
    ax.set_title(
        f"Definitive Sensitivity Ranking  —  W*_total = {W_opt/G:.3f} kg\n"
        f"payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}",
        fontsize=14)
    ax.legend(handles=[
        Patch(fc=TYPE_COLORS[t], label=lb) for t, lb in [
            ("input","Input / mission"), ("param","Parametric"),
            ("model","Model-form")]],
        fontsize=11, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.25)
    ax.axvline(0.05, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    fname = os.path.join(
        RESULTS_DIR,
        f"totals_ranking_{PAYLOAD_KG:g}_{N_C}_{RANGE_KM}km.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"  → {fname}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# UMDO PARAMETER SELECTION GUIDANCE
# ────────────────────────────────────────────────────────────────────────────
def umdo_guidance(all_results: list[dict]) -> None:
    def _best_s(r):
        return r["S_analytic"] if r.get("S_analytic") is not None \
               else r.get("S_fd", 0.0)

    ranked = sorted(all_results, key=lambda r: abs(_best_s(r)), reverse=True)

    print("\n── UMDO parameter selection (analytic + FD validated) ───────────")
    print("  S = |(dW*/dp)·(p_nom/W*)| — normalised elasticity at optimum\n")
    print("  Top-3 candidates:")
    for i, r in enumerate(ranked[:3], 1):
        s = _best_s(r)
        src = "analytic" if r.get("S_analytic") is not None else "FD"
        print(f"    {i}. {r['label']:30s}  |S|={abs(s):.4f}  [{r['type']}]  ({src})")

    freeze = [r for r in ranked if abs(_best_s(r)) < 0.02]
    if freeze:
        print(f"\n  Safe to freeze ({len(freeze)} with |S| < 0.02):")
        for r in freeze:
            print(f"    – {r['label']:30s}  |S|={abs(_best_s(r)):.4f}")

    unvalidated = [r for r in ranked
                   if r.get("S_analytic") is not None and not r.get("agreed")]
    if unvalidated:
        print(f"\n  ⚠  Analytic/FD mismatch — check component partials for:")
        for r in unvalidated:
            print(f"    ✗ {r['label']:30s}  "
                  f"S_analytic={r['S_analytic']:+.4f}  "
                  f"S_FD={r.get('S_fd', float('nan')):+.4f}  "
                  f"rel_err={r.get('rel_err', 0):.2%}")
        print("    Use prob.check_partials() to locate the incorrect partial.")

    print(f"\n  → 1-param UMDO: {ranked[0]['label']}")
    if len(ranked) > 1:
        print(f"  → 2-param UMDO: {ranked[0]['label']}  +  {ranked[1]['label']}")
    print("─────────────────────────────────────────────────────────────────\n")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── QBiT Analytic Sensitivity Analysis (compute_totals) ──────────")
    print(f"   Mission: payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print("─────────────────────────────────────────────────────────────────\n")

    # Step 1 — run deterministic optimisation
    print("Step 1 / 4  —  Deterministic optimisation …")
    prob  = build_and_optimise()
    W_opt = float(prob.get_val("W_total")[0])
    print(f"  W*_total = {W_opt/G:.4f} kg\n")

    # Step 2 — list inputs if requested, then exit
    if args.list_inputs:
        list_inputs(prob)
        print("  → Fill in om_name column in PARAMS, then re-run without --list-inputs.")
        raise SystemExit(0)

    # Step 3 — analytic total derivatives
    print("Step 2 / 4  —  Analytic total derivatives (compute_totals) …\n")
    analytic_results = compute_analytic_sensitivities(prob, W_opt)

    # Step 4 — FD cross-validation of analytic results
    print("\nStep 3 / 4  —  FD cross-validation (re-optimise at "
          f"±{FD_FRAC:.1%}) …")
    if analytic_results:
        validated = fd_crossvalidation(analytic_results, W_opt)
    else:
        validated = []
        print("  [SKIP] No analytic results to validate.")

    # Step 5 — FD-only for unwired parameters
    fd_results = fd_only_sensitivity(W_opt)
    all_results = validated + fd_results

    # Report
    print("\nStep 4 / 4  —  Results\n")
    print_table(all_results, W_opt)

    print("\nGenerating plots …")
    if validated:
        plot_comparison(validated, W_opt)
    plot_ranking(all_results, W_opt)

    umdo_guidance(all_results)