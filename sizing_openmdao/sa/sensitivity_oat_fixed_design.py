"""
sensitivity_oat_fixed_design.py  —  OAT sensitivities at a fixed design

This script duplicates the correct OAT approach from `sensitivity_oat.py`
but evaluates parameter perturbations at a fixed design point (fixed
`V_inf`, `r`, `J`, `S_w`). For each perturbation we:
 - patch `qbit.constants` with the perturbed value
 - reload qbit submodules and rebuild the Problem so components pick up
   the perturbed constants at import time
 - set the design variables as input defaults
 - solve the model-consistent `W_total` by root-finding the `weight_residual`

This preserves the scientific interpretation: we are measuring how the
model's predicted MTOM at a fixed design changes with param perturbations
(i.e. NOT re-optimising the design per perturbation).

Usage:
    python sensitivity_oat_fixed_design.py --use-user-fixed
    python sensitivity_oat_fixed_design.py --payload 3 --range 30 --nc 3
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import importlib
import warnings
import numpy as np
import csv

# PATH: allow importing qbit from parent folder
here = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(here, '..'))
sys.path.append(parent_dir)

import openmdao.api as om
import qbit.constants as C
from qbit.constants import G
from sensitivity_oat import PARAMS, PERTURB_FRACS, _apply_perturbation, _restore_all
from qbit.models.qbit_model import build_qbit_model


# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--payload", type=float, default=3.0)
parser.add_argument("--range",   type=float, default=15.0,  help="km")
parser.add_argument("--nc",      type=int,   default=2)
# Note: the script always computes the optimized baseline design by default.
# The previous `--use-user-fixed` mode and manual `USER_FIXEDS` block were
# removed to avoid requiring manual fixed-design entry on each run.
parser.add_argument("--outdir", type=str, default=None, help="Output directory for CSV and prints")
args, _ = parser.parse_known_args()

PAYLOAD_KG = args.payload
RANGE_M = args.range * 1_000.0
N_C = args.nc
RANGE_KM = int(RANGE_M / 1_000.0)
PAYLOAD_STR = f"{PAYLOAD_KG:g}"

RESULTS_DIR = args.outdir or os.path.join(parent_dir, "sa", f"results_fixed_from_oat_{PAYLOAD_STR}_{N_C}_{RANGE_KM}")
os.makedirs(RESULTS_DIR, exist_ok=True)


# (No manual fixed-design defaults — optimized design is computed automatically.)


def compute_optimized_design(payload_kg: float, range_m: float, n_c: int) -> dict:
    """Run the optimiser once to extract a reference design (optional)."""
    # reload non-constants qbit modules so they pick up C.* values
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass

    try:
        from qbit.models.qbit_model import build_qbit_model as _build_qbit_model
    except Exception:
        _build_qbit_model = build_qbit_model

    prob = om.Problem(reports=None)
    prob.model = _build_qbit_model(payload_kg, range_m, n_c)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["maxiter"] = 2000

    prob.model.add_design_var("W_total")
    prob.model.add_design_var("V_inf")
    prob.model.add_design_var("r")
    prob.model.add_design_var("J")
    prob.model.add_design_var("S_w")

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading",    upper=C.DL_MAX)
    prob.model.add_constraint("blade_loading",   upper=C.BL_MAX)
    prob.model.add_constraint("cruise_CL",       upper=C.CL_MAX)

    try:
        prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    except Exception:
        pass

    prob.setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()

    vals = {}
    for name in ("V_inf", "r", "J", "S_w", "W_total"):
        try:
            v = float(prob.get_val(name)[0])
        except Exception:
            try:
                v = float(prob.get_val(name))
            except Exception:
                v = None
        vals[name] = v
    return vals


def build_and_solve_W(design_vals: dict, payload_kg: float, range_m: float, n_c: int) -> float:
    """Build a Problem with the current qbit.constants and fixed design inputs,
    then find the model-consistent W_total by root-finding weight_residual."""
    # ensure modules that import constants are reloaded so they pick up C.* values
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass

    try:
        from qbit.models.qbit_model import build_qbit_model as _build_qbit_model
    except Exception:
        _build_qbit_model = build_qbit_model

    prob = om.Problem(reports=None)
    prob.model = _build_qbit_model(payload_kg, range_m, n_c)

    # set design inputs so components see the fixed design
    try:
        if "V_inf" in design_vals:
            prob.model.set_input_defaults("V_inf", val=design_vals["V_inf"], units="m/s")
        if "r" in design_vals:
            prob.model.set_input_defaults("r", val=design_vals["r"], units="m")
        if "S_w" in design_vals:
            prob.model.set_input_defaults("S_w", val=design_vals["S_w"], units="m**2")
        if "J" in design_vals:
            prob.model.set_input_defaults("J", val=design_vals["J"])  # unitless
        # benign default to remove promoted-input ambiguity; we'll overwrite W during root find
        prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    except Exception:
        pass

    prob.setup()

    # optional debug: print current constants imported into a representative component
    if os.getenv("SENS_OAT_DEBUG") == "1":
        try:
            vals = {attr: getattr(C, attr, '<missing>') for attr, _, _, _, _ in PARAMS}
            print(f"[DEBUG] qbit.constants (after reload): {vals}")
        except Exception:
            pass

    def _residual_for_w(w_val: float) -> float:
        try:
            prob.set_val("W_total", w_val)
        except Exception:
            prob.set_val("W_total", w_val, indices=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_model()
        try:
            rr = prob.get_val("weight_residual")
        except Exception:
            rr = prob.get_val("weight_residual")
        if os.getenv("SENS_OAT_DEBUG") == "1":
            try:
                wt = prob.get_val("W_total")
                print(f"[DEBUG] run_model W_total={wt} weight_residual={rr}")
            except Exception:
                pass
        try:
            return float(rr[0])
        except Exception:
            return float(rr)

    w_lo = 3.0 * G
    w_hi = 60.0 * G
    r_lo = _residual_for_w(w_lo)
    r_hi = _residual_for_w(w_hi)

    # expand if bracket not found
    max_expand = 6
    expand_i = 0
    while r_lo * r_hi > 0 and expand_i < max_expand:
        w_hi *= 2.0
        r_hi = _residual_for_w(w_hi)
        expand_i += 1

    if r_lo * r_hi > 0:
        # fallback: best-effort single run
        try:
            prob.set_val("W_total", w_lo)
            prob.run_model()
            return float(prob.get_val("W_total")[0])
        except Exception:
            return 6.0 * G

    w_mid = w_lo
    for _ in range(60):
        w_mid = 0.5 * (w_lo + w_hi)
        r_mid = _residual_for_w(w_mid)
        if abs(r_mid) < 1e-6:
            break
        if r_lo * r_mid <= 0:
            w_hi = w_mid
            r_hi = r_mid
        else:
            w_lo = w_mid
            r_lo = r_mid

    try:
        prob.set_val("W_total", w_mid)
        prob.run_model()
        W = float(prob.get_val("W_total")[0])
    except Exception:
        try:
            W = float(prob.get_val("W_total"))
        except Exception:
            W = w_mid
    return W


def main():
    _restore_all()

    print("Computing optimized baseline design (one full optimisation)...")
    t0 = time.time()
    design_full = compute_optimized_design(PAYLOAD_KG, RANGE_M, N_C)
    print(f"  Optimisation done in {time.time() - t0:.1f}s")
    design = {k: design_full[k] for k in ("V_inf", "r", "S_w", "J")}
    print("Using optimized design values:")
    for k, v in design.items():
        print(f"  {k}: {v}")

    _restore_all()
    # compute baseline W_total: either take from full optimisation (if used),
    # or compute by optimising only W_total when user-provided fixed design requested
    def optimise_W_for_fixed_design(design_vals: dict, payload_kg: float, range_m: float, n_c: int) -> dict:
        for name in list(sys.modules.keys()):
            if name.startswith("qbit.") and name != "qbit.constants":
                try:
                    importlib.reload(sys.modules[name])
                except Exception:
                    pass
        try:
            from qbit.models.qbit_model import build_qbit_model as _build_qbit_model
        except Exception:
            _build_qbit_model = build_qbit_model

        prob = om.Problem(reports=None)
        prob.model = _build_qbit_model(payload_kg, range_m, n_c)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.options["tol"] = 1e-9
        prob.driver.options["maxiter"] = 2000

        # only W_total free; fix other design vars by setting their defaults and not adding them
        prob.model.add_design_var("W_total")
        prob.model.add_objective("W_total")
        prob.model.add_constraint("weight_residual", equals=0.0)
        prob.model.add_constraint("disk_loading",    upper=om.Problem if False else None) if False else None

        # set fixed design inputs
        try:
            if "V_inf" in design_vals:
                prob.model.set_input_defaults("V_inf", val=design_vals["V_inf"], units="m/s")
            if "r" in design_vals:
                prob.model.set_input_defaults("r", val=design_vals["r"], units="m")
            if "S_w" in design_vals:
                prob.model.set_input_defaults("S_w", val=design_vals["S_w"], units="m**2")
            if "J" in design_vals:
                prob.model.set_input_defaults("J", val=design_vals["J"])  # unitless
            prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
        except Exception:
            pass

        prob.setup()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.run_driver()

        try:
            W_val = float(prob.get_val("W_total")[0])
        except Exception:
            W_val = float(prob.get_val("W_total"))

        # Check key constraints and report violations relative to current limits
        violations = {}
        try:
            dl = float(np.atleast_1d(prob.get_val("disk_loading"))[0])
            violations["disk_loading"] = (dl, float(getattr(C, "DL_MAX", float('inf'))), dl > getattr(C, "DL_MAX", float('inf')))
        except Exception:
            violations["disk_loading"] = (None, getattr(C, "DL_MAX", None), False)
        try:
            bl = float(np.atleast_1d(prob.get_val("blade_loading"))[0])
            violations["blade_loading"] = (bl, float(getattr(C, "BL_MAX", float('inf'))), bl > getattr(C, "BL_MAX", float('inf')))
        except Exception:
            violations["blade_loading"] = (None, getattr(C, "BL_MAX", None), False)
        try:
            cl = float(np.atleast_1d(prob.get_val("cruise_CL"))[0])
            violations["cruise_CL"] = (cl, float(getattr(C, "CL_MAX", float('inf'))), cl > getattr(C, "CL_MAX", float('inf')))
        except Exception:
            violations["cruise_CL"] = (None, getattr(C, "CL_MAX", None), False)

        return {"W": W_val, "violations": violations}

    # Always compute baseline W_total using the robust single-variable optimiser
    # (do not trust the raw value returned by the potentially-failing full run).
    W_base_res = optimise_W_for_fixed_design(design, PAYLOAD_KG, RANGE_M, N_C)
    W_base = W_base_res["W"] if isinstance(W_base_res, dict) else float(W_base_res)

    print(f"Baseline W_total at fixed design = {W_base/G:.6f} kg\n")

    results = []
    n_total = len(PARAMS) * len(PERTURB_FRACS)
    run_count = 0
    t0 = time.time()

    for attr, nominal, label, type_tag, perturb_abs in PARAMS:
        S_list, dW_list, W_list = [], [], []
        margins_list = []
        any_fail = False
        for frac in PERTURB_FRACS:
            run_count += 1
            elapsed = time.time() - t0
            eta = (elapsed / run_count) * (n_total - run_count) if run_count > 1 else 0.0
            print(f"  [{run_count:>3}/{n_total}] {label:28s} Δ={frac:+.0%}  ETA {eta:.0f}s …", end="\r", flush=True)

            _restore_all()
            p_val = _apply_perturbation(attr, nominal, frac, perturb_abs)

            try:
                W_eval_res = optimise_W_for_fixed_design(design, PAYLOAD_KG, RANGE_M, N_C)
                if isinstance(W_eval_res, dict):
                    W_eval = float(W_eval_res.get("W", float('nan')))
                    violations = W_eval_res.get("violations", {})
                else:
                    W_eval = float(W_eval_res)
                    violations = {}
            except Exception:
                any_fail = True
                W_eval = float('nan')
                violations = {}

            dW = 100.0 * (W_eval - W_base) / W_base if W_base != 0 else float('nan')
            actual_frac = (p_val - nominal) / abs(nominal) if nominal != 0 else frac
            S = dW / (actual_frac * 100.0) if actual_frac != 0 else 0.0

            S_list.append(S)
            dW_list.append(dW)
            W_list.append(W_eval)

            # compute normalized margins: (limit - value)/limit -> negative means violated
            m = {}
            for k, vv in violations.items():
                val, limit, _ = (None, None, False) if vv is None else vv
                try:
                    if val is None or limit is None:
                        m[k] = None
                    else:
                        m[k] = (limit - float(val)) / float(limit) if float(limit) != 0 else None
                except Exception:
                    m[k] = None
            margins_list.append(m)
            # If any constraint was violated, flag and print a concise warning
            vio_flag = False
            for k, v in violations.items():
                if isinstance(v, tuple) and len(v) == 3 and v[2]:
                    vio_flag = True
                    print(f"\n[VIOLATION] {label}: {k}={v[0]:.4g} > limit={v[1]:.4g}")
            if vio_flag:
                any_fail = True

            # Print normalized margins to terminal for quick inspection
            try:
                parts = []
                for kk, mv in m.items():
                    if mv is None:
                        parts.append(f"{kk}:NA")
                    else:
                        parts.append(f"{kk}:{mv*100:+.1f}%")
                mstr = ", ".join(parts)
                print(f"\n[MARGINS] {label}: {mstr}")
            except Exception:
                pass

        _restore_all()

        S_mean = float(np.mean(np.abs(S_list)))
        results.append({
            "attr": attr,
            "label": label,
            "type": type_tag,
            "nominal": nominal,
            "S_mean": S_mean,
            "S_vals": S_list,
            "dW_pcts": dW_list,
            "W_opts": W_list,
            "failed": any_fail,
            "margins_list": margins_list,
        })

    results.sort(key=lambda r: r["S_mean"], reverse=True)

    # For any parameter that showed infeasibility in the discrete sweep,
    # find the smallest perturbation fraction that causes a violation via bisection.
    def _param_perturb_abs(attr_name):
        for a, nom, lab, t, pabs in PARAMS:
            if a == attr_name:
                return pabs
        return False

    def _check_violation(attr_name, nominal, frac) -> tuple[bool, dict]:
        """Apply perturbation and return (is_violated, violations_dict)."""
        _restore_all()
        p_abs = _param_perturb_abs(attr_name)
        _apply_perturbation(attr_name, nominal, frac, p_abs)
        res = optimise_W_for_fixed_design(design, PAYLOAD_KG, RANGE_M, N_C)
        _restore_all()
        if isinstance(res, dict):
            vio = res.get("violations", {})
            any_v = any(isinstance(v, tuple) and len(v) == 3 and v[2] for v in vio.values())
            return any_v, vio
        return False, {}

    print("\nFixed-design infeasibility summary (tipping-point per parameter):")
    for r in results:
        margins = r.get("margins_list", [])
        # find first discrete index where any margin < 0
        idx = None
        for i, mm in enumerate(margins):
            if not mm:
                continue
            if any((v is not None and v < 0) for v in mm.values()):
                idx = i
                break
        if idx is None:
            continue

        # bracket: low -> last non-violating (or 0.0), high -> PERTURB_FRACS[idx]
        high = PERTURB_FRACS[idx]
        low = 0.0
        # if previous index exists and was non-violating, use that as low
        if idx > 0:
            low = PERTURB_FRACS[idx - 1]

        # bisection in fraction space
        tol = 1e-3
        max_iter = 25
        hi_viol, hi_vio = _check_violation(r["attr"], r["nominal"], high)
        lo_viol, lo_vio = _check_violation(r["attr"], r["nominal"], low)
        if not hi_viol:
            # unexpected: discrete high didn't violate; skip
            r["tipping_frac"] = None
            r["tipping_constraint"] = None
            r["tipping_margin_pct"] = None
            continue
        if lo_viol:
            # violation already at low bound (likely low==0); record high
            r["tipping_frac"] = high
            # pick worst constraint at high
            worst_k = None
            worst_m = None
            for k, vv in hi_vio.items():
                if isinstance(vv, tuple) and vv[2]:
                    val, limit, _ = vv
                    mval = (limit - float(val)) / float(limit) if limit else None
                    if worst_m is None or (mval is not None and mval < worst_m):
                        worst_m = mval
                        worst_k = k
            r["tipping_constraint"] = worst_k
            r["tipping_margin_pct"] = (worst_m * 100.0) if worst_m is not None else None
            continue

        lo = float(low)
        hi = float(high)
        iter_i = 0
        last_vio = hi_vio
        while abs(hi - lo) > tol and iter_i < max_iter:
            mid = 0.5 * (lo + hi)
            vio_mid, vio_dict = _check_violation(r["attr"], r["nominal"], mid)
            if vio_mid:
                hi = mid
                last_vio = vio_dict
            else:
                lo = mid
            iter_i += 1

        r["tipping_frac"] = hi
        # select worst violated constraint from last_vio
        worst_k = None
        worst_m = None
        for k, vv in (last_vio or {}).items():
            if isinstance(vv, tuple) and vv[2]:
                val, limit, _ = vv
                try:
                    mval = (limit - float(val)) / float(limit) if limit else None
                except Exception:
                    mval = None
                if worst_m is None or (mval is not None and mval < worst_m):
                    worst_m = mval
                    worst_k = k
        r["tipping_constraint"] = worst_k
        r["tipping_margin_pct"] = (worst_m * 100.0) if worst_m is not None else None
        print(f"  {r['label']:30s} tipping at {r['tipping_frac']:+.2%} -> {r['tipping_constraint']} margin={r['tipping_margin_pct']:+.1f}%")
    print("")

    outcsv = os.path.join(RESULTS_DIR, f"sensitivity_fixed_from_oat_ranked_payload{PAYLOAD_STR}_nc{N_C}_R{RANGE_KM}km.csv")
    with open(outcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attr", "label", "type", "nominal", "S_mean", "dW_minus20_pct", "dW_minus10_pct", "dW_minus5_pct", "dW_plus5_pct", "dW_plus10_pct", "dW_plus20_pct", "failed", "tipping_frac", "tipping_constraint", "tipping_margin_pct"])
        for i, r in enumerate(results, 1):
            dw = r["dW_pcts"]
            tip_frac = r.get("tipping_frac")
            tip_cons = r.get("tipping_constraint")
            tip_margin = r.get("tipping_margin_pct")
            w.writerow([i, r["attr"], r["label"], r["type"], r["nominal"], f"{r['S_mean']:.6f}",
                        f"{dw[0]:.6f}", f"{dw[1]:.6f}", f"{dw[2]:.6f}", f"{dw[3]:.6f}", f"{dw[4]:.6f}", f"{dw[5]:.6f}", r["failed"],
                        (f"{tip_frac:.6f}" if tip_frac is not None else ""), (tip_cons if tip_cons is not None else ""), (f"{tip_margin:.3f}" if tip_margin is not None else "")])

    print(f"\nResults written to: {outcsv}")

    # Print full table matching sensitivity_oat.py format
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
            dw = r['dW_pcts']
            flg = ' ◀◀' if r['S_mean'] >= 0.5 else (' ◀' if r['S_mean'] >= 0.1 else '')
            print(f"  {i:>2}  {r['label']:30s}  {r['type']:6s}  "
                  f"{r['S_mean']:>7.4f}  "
                  f"{dw[1]:>+8.2f}%  "
                  f"{dw[4]:>+8.2f}%  "
                  f"{dw[5]:>+8.2f}%{flg}")
        print(SEP)

    print_table(results, W_base)


if __name__ == '__main__':
    print("\n── Fixed-Design OAT (from sensitivity_oat) ─────────────────────")
    print(f"   Mission:  payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print("──────────────────────────────────────────────────────────────\n")
    start = time.time()
    main()
    print(f"\nDone — {time.time() - start:.1f}s total")