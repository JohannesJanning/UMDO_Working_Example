"""
sensitivity_screening.py  —  Morris (EE) + Sobol global sensitivity screening
                              for the QBiT UMDO parameter importance ranking.

PURPOSE (UMDO context)
──────────────────────
This script implements Step 2.2.1.2 of the UMDO process (Wang 2023 / Yao 2011):
    "Use sensitivity analysis to screen out factors which have no significant
     influence on system design, so as to simplify the UMDO problem."

The CORRECT evaluator is W* = re-optimised MTOM, NOT a fixed-design root-find.
Each model call runs the full SLSQP optimiser — exactly like sensitivity_oat.py.
This answers: "which uncertain parameters shift the *optimal* MTOM the most,
across the full uncertain space?"

METHODS
──────────────────────
Morris (Elementary Effects):
  - Cheap: R*(k+1) optimiser calls, typically R=15, k=21 → ~330 runs
  - Outputs: μ* (importance), σ (nonlinearity / interaction)
  - Use for: initial screening / freeze decisions
  - Reference: Morris (1991), Campolongo et al. (2007)

Sobol (variance-based):
  - Expensive: N*(2k+2) optimiser calls, typically N=64 → ~2752 runs
  - Outputs: S1 (first-order), ST (total-order) indices
  - Use for: quantitative importance ranking, UQ propagation input selection
  - Reference: Saltelli et al. (2010), Sobol (1993)
  - Requires: pip install SALib

RUN MODES
──────────────────────
  python sensitivity_screening.py                    # Morris only (fast)
  python sizing_openmdao/sa/sensitivity_screening.py --method sobol     # Sobol only
  python sizing_openmdao/sa/sensitivity_screening.py --method both      # Morris then Sobol
  python sizing_openmdao/sa/sensitivity_screening.py --method morris --r 20   # 20 trajectories
  python sizing_openmdao/sa/sensitivity_screening.py --method sobol  --n 256  # 256 Sobol samples
  python sizing_openmdao/sa/sensitivity_screening.py --payload 3 --range 25 --nc 5

RUNTIME ESTIMATE (per optimiser call ≈ 0.3 s on a laptop)
  Morris  R=15, k=21 → 336 calls → ~100 s
  Sobol   N=64, k=21 → 2752 calls → ~14 min
  Sobol   N=128      → 5504 calls → ~28 min

IMPORTANT: bounds in PARAMS below define the uncertainty space.
  Adjust them to match your uncertainty quantification (±10%, ±20%, physical
  limits, or literature-based CoV). The ranking is sensitive to these bounds.

  
USE THE FOLLOWING CODE FOR SNIPPET FOR % PERTURBATION 


def bounds(nominal, pct, lb_hard=None, ub_hard=None):
    lb = nominal * (1.0 - pct)
    ub = nominal * (1.0 + pct)
    if lb_hard is not None: lb = max(lb, lb_hard)
    if ub_hard is not None: ub = min(ub, ub_hard)
    return lb, ub

PCT = 0.20  # ±20% uniform perturbation — matches OAT sweep

PARAMS = [
    # attr              nominal    label                    type     *bounds(nominal, PCT, hard_lb, hard_ub)
    ("RHO_AIR",         1.225,    "ρ_air",                "input", *bounds(1.225,    PCT)            ),
    ("T_HOVER",         60.0,     "t_hover",              "input", *bounds(60.0,     PCT, 1.0)       ),  # lb>0
    ("BETA_QBIT",       0.18,     "β_QBiT (frame frac.)", "param", *bounds(0.18,     PCT, 0.0, 1.0) ),
    ("ETA_HOVER",       0.65,     "η_hover",              "param", *bounds(0.65,     PCT, 0.0, 1.0) ),
    ("CD0_WING",        0.01,     "CD0_wing",             "param", *bounds(0.01,     PCT, 0.0)       ),
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", *bounds(0.80,     PCT, 0.0, 1.0) ),
    ("AR_FIXED",        8.0,      "AR",                   "input", *bounds(8.0,      PCT, 1.0)       ),
    ("SIGMA",           0.13,     "σ (solidity)",         "param", *bounds(0.13,     PCT, 0.0)       ),
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", *bounds(0.012,    PCT, 0.0)       ),
    ("KAPPA_MAX",       1.15,     "κ_max",                "param", *bounds(1.15,     PCT, 1.0)       ),  # κ≥1 physically
    ("BATTERY_DENSITY", 158.0,    "ρ_bat (Wh/kg)",        "input", *bounds(158.0,    PCT, 1.0)       ),
    ("BATTERY_EFF",     0.85,     "η_bat",                "param", *bounds(0.85,     PCT, 0.0, 1.0) ),
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", *bounds(2.506e-4, PCT, 0.0)       ),
    ("K_ESC",           3.594e-4, "k_ESC",                "model", *bounds(3.594e-4, PCT, 0.0)       ),
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", *bounds(0.7484,   PCT)            ),
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", *bounds(0.0403,   PCT, 0.0)       ),
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", *bounds(-0.0802,  PCT)            ),  # negative nominal, symmetric ok
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", *bounds(2.2854,   PCT, 0.0)       ),
    ("DL_MAX",          55.0,     "DL_MAX (disk load.)",  "param", *bounds(55.0,     PCT, 0.0)       ),
    ("BL_MAX",          80.0,     "BL_MAX (blade load.)", "param", *bounds(80.0,     PCT, 0.0)       ),
    ("CL_MAX",          1.0,      "CL_MAX (cruise CL)",   "param", *bounds(1.0,      PCT, 0.0, 2.0) ),
]

PARAMS = [
    # attr              nominal    label                    type     lb          ub          justification
    ("RHO_AIR",         1.225,    "ρ_air",                "input", 1.040,      1.410),     # ±15%: ISA 0-2000m + temp variation
    ("T_HOVER",         60.0,     "t_hover",              "input", 30.0,       180.0),     # physical mission envelope, not %-based
    ("BETA_QBIT",       0.18,     "β_QBiT (frame frac.)", "param", 0.144,      0.216),     # ±20%: Raymer conceptual stage structural uncertainty
    ("ETA_HOVER",       0.65,     "η_hover",              "param", 0.550,      0.750),     # literature range small UAV rotors (Leishman 2006)
    ("CD0_WING",        0.01,     "CD0_wing",             "param", 0.008,      0.012),     # ±20%: conceptual-stage CFD/panel scatter
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", 0.680,      0.920),     # ±15%: panel method uncertainty
    ("AR_FIXED",        8.0,      "AR",                   "input", 6.4,        9.6),       # ±20%: design space exploration
    ("SIGMA",           0.13,     "σ (solidity)",         "param", 0.104,      0.156),     # ±20%: manufacturing tolerance + design choice
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", 0.009,      0.015),     # ±20%: CFD/BEM scatter at conceptual stage
    ("KAPPA_MAX",       1.15,     "κ_max",                "param", 1.035,      1.265),     # ±10%: well-characterised inflow correction
    ("BATTERY_DENSITY", 158.0,    "ρ_bat (Wh/kg)",        "input", 126.0,      200.0),     # LiPo technology range 2024 (manufacturer data)
    ("BATTERY_EFF",     0.85,     "η_bat",                "param", 0.720,      0.980),     # ±15%, capped below 1.0: C-rate + temp effects
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", 2.130e-4,   2.882e-4),  # ±15%: regression 95% prediction interval
    ("K_ESC",           3.594e-4, "k_ESC",                "model", 3.055e-4,   4.133e-4),  # ±15%: regression 95% prediction interval
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", 0.636,      0.861),     # ±15%: regression 95% prediction interval
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", 0.034,      0.046),     # ±15%: regression 95% prediction interval
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", -0.092,    -0.068),     # ±15%: regression 95% prediction interval
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", 1.943,      2.628),     # ±15%: regression 95% prediction interval
    ("DL_MAX",          55.0,     "DL_MAX (disk load.)",  "param", 44.0,       66.0),      # ±20%: noise regs + structural assumption uncertainty
    ("BL_MAX",          80.0,     "BL_MAX (blade load.)", "param", 64.0,       96.0),      # ±20%: blade stall characterisation uncertainty
    ("CL_MAX",          1.0,      "CL_MAX (cruise CL)",   "param", 0.800,      1.200),     # ±20%: airfoil + flap assumption uncertainty
]
"""

from __future__ import annotations
import argparse
import os
import sys
import importlib
import warnings
import time
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path fix ─────────────────────────────────────────────────────────────────
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import openmdao.api as om
import qbit.constants as C
from qbit.constants import G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS

# ── matplotlib style (matches sensitivity_oat.py) ────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 1.8, "lines.markersize": 6,
})

TYPE_COLORS = {
    "input": "#388bfd",
    "param": "#d29922",
    "model": "#a371f7",
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--payload", type=float, default=3.0)
parser.add_argument("--range",   type=float, default=15.0, help="km one-way")
parser.add_argument("--nc",      type=int,   default=2)
parser.add_argument("--method",  type=str,   default="morris",
                    choices=["morris", "sobol", "both"],
                    help="Screening method (default: morris)")
parser.add_argument("--r",       type=int,   default=15,
                    help="Morris: number of trajectories (default 15)")
parser.add_argument("--p",       type=int,   default=6,
                    help="Morris: number of grid levels (default 6)")
parser.add_argument("--n",       type=int,   default=64,
                    help="Sobol: base sample count N (total runs = N*(2k+2))")
parser.add_argument("--seed",    type=int,   default=42)
args, _ = parser.parse_known_args()

PAYLOAD_KG  = args.payload
RANGE_M     = args.range * 1_000.0
N_C         = args.nc
METHOD      = args.method
R_MORRIS    = args.r
P_MORRIS    = args.p
N_SOBOL     = args.n
SEED        = args.seed
RANGE_KM    = int(RANGE_M / 1_000)
PAYLOAD_STR = f"{PAYLOAD_KG:g}"

RESULTS_DIR = os.path.join(
    parent_dir, "sa",
    f"screening_{PAYLOAD_STR}_{N_C}_{RANGE_KM}_{METHOD}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── PARAMETER TABLE ───────────────────────────────────────────────────────────
# Columns: (const_attr, nominal, label, type_tag, lower_bound, upper_bound)
#
# Bounds define the UNCERTAINTY SPACE for global screening.
# These are ±20% of nominal unless a physical constraint applies.
# Adjust to match your epistemic uncertainty characterisation.
#
# type_tag: "input"  → mission / environment input (always uncertain)
#           "param"  → aerodynamic / physical parameter
#           "model"  → regression / empirical model coefficient
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = [
    # attr              nominal    label                    type     lb          ub          justification
    ("RHO_AIR",         1.225,    "ρ_air",                "input", 1.040,      1.410),     # ±15%: ISA 0-2000m + temp variation
    ("T_HOVER",         60.0,     "t_hover",              "input", 30.0,       180.0),     # physical mission envelope, not %-based
    ("BETA_QBIT",       0.18,     "β_QBiT (frame frac.)", "param", 0.144,      0.216),     # ±20%: Raymer conceptual stage structural uncertainty
    ("ETA_HOVER",       0.65,     "η_hover",              "param", 0.550,      0.750),     # literature range small UAV rotors (Leishman 2006)
    ("CD0_WING",        0.01,     "CD0_wing",             "param", 0.008,      0.012),     # ±20%: conceptual-stage CFD/panel scatter
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", 0.680,      0.920),     # ±15%: panel method uncertainty
    ("AR_FIXED",        8.0,      "AR",                   "input", 6.4,        9.6),       # ±20%: design space exploration
    ("SIGMA",           0.13,     "σ (solidity)",         "param", 0.104,      0.156),     # ±20%: manufacturing tolerance + design choice
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", 0.009,      0.015),     # ±20%: CFD/BEM scatter at conceptual stage
    ("KAPPA_MAX",       1.15,     "κ_max",                "param", 1.035,      1.265),     # ±10%: well-characterised inflow correction
    ("BATTERY_DENSITY", 158.0,    "ρ_bat (Wh/kg)",        "input", 126.0,      200.0),     # LiPo technology range 2024 (manufacturer data)
    ("BATTERY_EFF",     0.85,     "η_bat",                "param", 0.720,      0.980),     # ±15%, capped below 1.0: C-rate + temp effects
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", 2.130e-4,   2.882e-4),  # ±15%: regression 95% prediction interval
    ("K_ESC",           3.594e-4, "k_ESC",                "model", 3.055e-4,   4.133e-4),  # ±15%: regression 95% prediction interval
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", 0.636,      0.861),     # ±15%: regression 95% prediction interval
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", 0.034,      0.046),     # ±15%: regression 95% prediction interval
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", -0.092,    -0.068),     # ±15%: regression 95% prediction interval
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", 1.943,      2.628),     # ±15%: regression 95% prediction interval
    ("DL_MAX",          55.0,     "DL_MAX (disk load.)",  "param", 44.0,       66.0),      # ±20%: noise regs + structural assumption uncertainty
    ("BL_MAX",          80.0,     "BL_MAX (blade load.)", "param", 64.0,       96.0),      # ±20%: blade stall characterisation uncertainty
    ("CL_MAX",          1.0,      "CL_MAX (cruise CL)",   "param", 0.800,      1.200),     # ±20%: airfoil + flap assumption uncertainty
]

k = len(PARAMS)
LB = np.array([p[4] for p in PARAMS])
UB = np.array([p[5] for p in PARAMS])


# ── CONSTANTS HELPERS (mirrors sensitivity_oat.py) ────────────────────────────
def _restore_all() -> None:
    for attr, nominal, *_ in PARAMS:
        setattr(C, attr, nominal)


def _set_constants_from_array(x: np.ndarray) -> None:
    """Set all constants from a parameter vector x (physical units)."""
    for j, (attr, *_) in enumerate(PARAMS):
        setattr(C, attr, float(x[j]))


def _reload_qbit() -> None:
    """Reload all qbit submodules (except constants) to pick up patched values."""
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass


# ── EVALUATOR: re-optimise W* (THE CORRECT EVALUATOR FOR UMDO SCREENING) ──────
def optimise_W(payload_kg: float, range_m: float, n_c: int) -> float:
    """
    Run the SLSQP optimiser with current C.* constants and return W*_total [N].

    This is the exact same function as sensitivity_oat.py::optimise(), keeping
    the design variables free so the design adapts to each parameter sample.
    The result is the *optimal* MTOM — the quantity whose sensitivity to
    uncertain parameters we are screening.
    """
    _reload_qbit()

    try:
        from qbit.models.qbit_model import build_qbit_model as _build
    except Exception:
        from qbit.models.qbit_model import build_qbit_model as _build

    prob = om.Problem(reports=None)
    prob.model = _build(payload_kg, range_m, n_c)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"]       = 1e-9
    prob.driver.options["maxiter"]   = 2000

    prob.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var("V_inf",   lower=V_INF_BOUNDS[0],   upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r",       lower=R_BOUNDS[0],        upper=R_BOUNDS[1])
    prob.model.add_design_var("J",       lower=J_BOUNDS[0],        upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w",     lower=S_W_BOUNDS[0],      upper=S_W_BOUNDS[1])

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading",    upper=C.DL_MAX)
    prob.model.add_constraint("blade_loading",   upper=C.BL_MAX)
    prob.model.add_constraint("cruise_CL",       upper=C.CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    prob.setup()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()

    return float(prob.get_val("W_total")[0])


def evaluate(x: np.ndarray) -> float:
    """
    Evaluate W* [N] for a parameter vector x in physical space.
    Sets all constants, reloads qbit modules, then optimises.
    Returns NaN on failure so outlier detection can handle it.
    """
    _restore_all()
    _set_constants_from_array(x)
    try:
        W = optimise_W(PAYLOAD_KG, RANGE_M, N_C)
        # Sanity check: reject obviously broken results
        if W < 1.0 * G or W > 500.0 * G:
            return float("nan")
        return W
    except Exception as e:
        return float("nan")
    finally:
        _restore_all()


# ── BASELINE ──────────────────────────────────────────────────────────────────
def compute_baseline() -> float:
    _restore_all()
    W = optimise_W(PAYLOAD_KG, RANGE_M, N_C)
    return W


# ═════════════════════════════════════════════════════════════════════════════
# MORRIS METHOD
# ═════════════════════════════════════════════════════════════════════════════

def _morris_trajectory(k: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one optimised Morris trajectory in [0,1]^k.

    Returns B* of shape (k+1, k). Successive rows differ in exactly one
    coordinate by ±Δ where Δ = p / (2*(p-1)).

    Reference: Morris (1991) Eq. 1-4; Campolongo (2007) improved sampling.
    """
    delta = p / (2.0 * (p - 1))

    # Random orientation matrix D* (diagonal ±1)
    D_star = np.diag(2 * rng.integers(0, 2, size=k) - 1)
    # Random permutation matrix P*
    perm = rng.permutation(k)
    P_star = np.eye(k, dtype=float)[perm]
    # Random base point x* in [0, 1-Δ]
    x_star = rng.uniform(0, 1.0 - delta, size=k)

    # Lower-triangular unit matrix
    B = np.tril(np.ones((k + 1, k), dtype=float), -1)
    J_col = np.ones((k + 1, 1))
    J_row = np.ones((1, k))

    B_star = (J_col @ x_star.reshape(1, k)
              + (delta / 2.0) * ((2.0 * B - J_col @ J_row) @ D_star @ P_star.T))
    # Clip to [0,1] to handle floating-point boundary issues
    return np.clip(B_star, 0.0, 1.0)


def _norm_to_physical(X_norm: np.ndarray) -> np.ndarray:
    """Scale from [0,1] to physical [lb, ub]."""
    return LB + X_norm * (UB - LB)


def run_morris(W_base: float) -> list[dict]:
    """
    Run Morris Elementary Effects screening.

    Returns list of dicts with keys:
        attr, label, type, nominal, mu_star, sigma, ee_vals
    """
    rng = np.random.default_rng(SEED)
    n_total = R_MORRIS * (k + 1)

    print(f"\n── Morris Elementary Effects ─────────────────────────────────────")
    print(f"   k={k} parameters, R={R_MORRIS} trajectories, p={P_MORRIS} levels")
    print(f"   Total optimiser calls: {n_total}")
    print(f"   Estimated runtime: {n_total * 0.3 / 60:.0f}–{n_total * 0.6 / 60:.0f} min\n")

    # ee_raw[j] = list of elementary effects for parameter j (in W* [N] / physical unit)
    ee_raw = [[] for _ in range(k)]
    n_fail = 0
    n_done = 0
    t0 = time.time()

    for tr in range(R_MORRIS):
        B_star = _morris_trajectory(k, P_MORRIS, rng)
        X_norm = B_star                           # already in [0,1]
        X_phys = _norm_to_physical(X_norm)        # physical values, shape (k+1, k)

        # Evaluate W* at each of the k+1 points in this trajectory
        Y = np.full(k + 1, np.nan)
        for i in range(k + 1):
            n_done += 1
            elapsed = time.time() - t0
            eta = (elapsed / n_done) * (n_total - n_done) if n_done > 1 else 0.0
            print(f"  [{n_done:>4}/{n_total}]  trajectory {tr+1:>2}/{R_MORRIS}  "
                  f"point {i+1:>2}/{k+1}  ETA {eta/60:.1f} min …",
                  end="\r", flush=True)
            Y[i] = evaluate(X_phys[i])
            if np.isnan(Y[i]):
                n_fail += 1

        # Compute elementary effects from this trajectory
        # The trajectory B_star has each step changing exactly one parameter.
        # We identify which parameter changed at each step by comparing rows.
        for step in range(k):
            diff_norm = X_norm[step + 1] - X_norm[step]
            j = int(np.argmax(np.abs(diff_norm)))    # which parameter moved
            delta_phys = X_phys[step + 1, j] - X_phys[step, j]
            if abs(delta_phys) < 1e-15:
                continue
            if np.isnan(Y[step]) or np.isnan(Y[step + 1]):
                continue
            EE = (Y[step + 1] - Y[step]) / delta_phys    # [N / physical_unit]
            ee_raw[j].append(EE)

    print(f"\n  Morris done: {n_done} evaluations, {n_fail} failures ({n_fail/n_done*100:.1f}%)")

    # Compute Morris statistics
    results = []
    for j, (attr, nominal, label, typ, lb, ub) in enumerate(PARAMS):
        vals = np.array(ee_raw[j])
        if len(vals) == 0:
            mu_star, sigma, ee_norm = 0.0, 0.0, []
        else:
            # Normalise: convert dimensional EE to elasticity
            # μ* = mean|EE| * (range_p / W_base) → dimensionless
            range_p = ub - lb
            ee_norm = [v * range_p / W_base for v in vals]
            mu_star = float(np.mean(np.abs(ee_norm)))
            sigma   = float(np.std(ee_norm))

        results.append({
            "attr": attr, "label": label, "type": typ, "nominal": nominal,
            "mu_star": mu_star, "sigma": sigma, "ee_vals": ee_norm,
            "n_ee": len(ee_norm),
        })

    results.sort(key=lambda r: r["mu_star"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SOBOL METHOD  (requires SALib)
# ═════════════════════════════════════════════════════════════════════════════

def _check_salib() -> bool:
    try:
        import SALib  # noqa: F401
        return True
    except ImportError:
        print("\n  [ERROR] SALib not installed. Run:  pip install SALib")
        print("  Skipping Sobol analysis.\n")
        return False


def run_sobol(W_base: float) -> list[dict]:
    if not _check_salib():
        return []

    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample

    from SALib.analyze import sobol as sobol_analyze

    n_total = N_SOBOL * (2 * k + 2)
    print(f"\n── Sobol Variance-Based Sensitivity ──────────────────────────────")
    print(f"   k={k} parameters, N={N_SOBOL} base samples")
    print(f"   Total optimiser calls: {n_total}")
    print(f"   Estimated runtime: {n_total * 0.3 / 60:.0f}–{n_total * 0.5 / 60:.0f} min\n")

    # Use unit hypercube for SALib — avoids ALL bound-sign issues.
    # We do our own affine scaling to physical space: x_phys = lb + u*(ub-lb)
    # Sobol indices are invariant to this transform.
    problem_unit = {
        "num_vars": k,
        "names":    [p[0] for p in PARAMS],
        "bounds":   [[0.0, 1.0]] * k,          # always legal, no sign issues
    }

    try:
        param_values_unit = sobol_sample.sample(
            problem_unit, N_SOBOL, calc_second_order=False, seed=SEED)
    except TypeError:
        np.random.seed(SEED)
        param_values_unit = sobol_sample.sample(
            problem_unit, N_SOBOL, calc_second_order=False)

    # Scale from [0,1] to physical [lb, ub]  — handles negative bounds fine
    param_values = LB + param_values_unit * (UB - LB)   # (n_samples, k)

    n_total_actual = len(param_values)
    Y = np.full(n_total_actual, np.nan)
    n_fail = 0
    t0 = time.time()

    for i, x in enumerate(param_values):
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n_total_actual - i - 1) if i > 0 else 0.0
        print(f"  [{i+1:>5}/{n_total_actual}]  ETA {eta/60:.1f} min …",
              end="\r", flush=True)
        Y[i] = evaluate(x)
        if np.isnan(Y[i]):
            n_fail += 1

    print(f"\n  Sobol evaluations done: {n_total_actual}, failures: {n_fail} "
          f"({n_fail/n_total_actual*100:.1f}%)")

    if n_fail > 0:
        mean_Y = np.nanmean(Y)
        Y = np.where(np.isnan(Y), mean_Y, Y)
        print(f"  [WARN] {n_fail} NaN results replaced with mean Y={mean_Y/G:.3f} kg")

    # Analyze against unit problem — indices are scale-invariant
    Si = sobol_analyze.analyze(problem_unit, Y,
                               calc_second_order=False,
                               print_to_console=False,
                               seed=SEED)

    results = []
    for j, (attr, nominal, label, typ, lb, ub) in enumerate(PARAMS):
        results.append({
            "attr": attr, "label": label, "type": typ, "nominal": nominal,
            "S1":      float(Si["S1"][j]),
            "S1_conf": float(Si["S1_conf"][j]),
            "ST":      float(Si["ST"][j]),
            "ST_conf": float(Si["ST_conf"][j]),
        })

    results.sort(key=lambda r: r["ST"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PRINT TABLES
# ═════════════════════════════════════════════════════════════════════════════

def print_morris_table(results: list[dict], W_base: float) -> None:
    SEP = "=" * 80
    print(f"\n{SEP}")
    print(f"  Morris (EE) Screening  —  W*_base = {W_base/G:.4f} kg  "
          f"[re-optimised evaluator]")
    print(f"  Mission: payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  "
          f"n_c={N_C}  |  R={R_MORRIS} trajectories, p={P_MORRIS} levels")
    print(SEP)
    print(f"  {'Rk':>2}  {'Parameter':30s}  {'Type':6s}  "
          f"{'μ*':>8}  {'σ':>8}  {'σ/μ*':>7}  Decision")
    print("  " + "─" * 72)
    for i, r in enumerate(results, 1):
        ratio = r["sigma"] / r["mu_star"] if r["mu_star"] > 1e-10 else 0.0
        if r["mu_star"] < 0.02:
            decision = "FREEZE"
        elif r["mu_star"] < 0.10:
            decision = "marginal"
        else:
            decision = "KEEP  ◀" + ("◀ nonlinear" if ratio > 1.0 else "")
        print(f"  {i:>2}  {r['label']:30s}  {r['type']:6s}  "
              f"{r['mu_star']:>8.4f}  {r['sigma']:>8.4f}  {ratio:>7.2f}  {decision}")
    print(SEP)
    print("  μ* = mean|EE| × (range_p / W*_base)  — normalised importance")
    print("  σ  = std(EE_norm)  — nonlinearity / parameter interactions")
    print("  σ/μ* > 1 → nonlinear or interacting with other parameters")
    print(f"{SEP}\n")


def print_sobol_table(results: list[dict], W_base: float) -> None:
    SEP = "=" * 88
    print(f"\n{SEP}")
    print(f"  Sobol Variance-Based Sensitivity  —  W*_base = {W_base/G:.4f} kg  "
          f"[re-optimised evaluator]")
    print(f"  Mission: payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  "
          f"n_c={N_C}  |  N={N_SOBOL} base samples")
    print(SEP)
    print(f"  {'Rk':>2}  {'Parameter':30s}  {'Type':6s}  "
          f"{'S1':>8}  {'±95%':>7}  {'ST':>8}  {'±95%':>7}  "
          f"{'ST-S1':>8}  Decision")
    print("  " + "─" * 82)
    for i, r in enumerate(results, 1):
        interaction = r["ST"] - r["S1"]
        if r["ST"] < 0.01:
            decision = "FREEZE"
        elif r["ST"] < 0.05:
            decision = "marginal"
        else:
            decision = "KEEP  ◀" + ("◀ interacts" if interaction > 0.05 else "")
        print(f"  {i:>2}  {r['label']:30s}  {r['type']:6s}  "
              f"{r['S1']:>8.4f}  {r['S1_conf']:>7.4f}  "
              f"{r['ST']:>8.4f}  {r['ST_conf']:>7.4f}  "
              f"{interaction:>8.4f}  {decision}")
    print(SEP)
    print("  S1 = first-order Sobol index  (main effect, fraction of total variance)")
    print("  ST = total-order Sobol index  (main + all interactions)")
    print("  ST - S1 > 0.05 → significant interaction with other parameters")
    print("  Use ST for UMDO parameter selection (includes interaction effects)")
    print(f"{SEP}\n")


# ═════════════════════════════════════════════════════════════════════════════
# UMDO GUIDANCE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def umdo_guidance(morris_results: list[dict], sobol_results: list[dict],
                  W_base: float) -> None:
    print("── UMDO Parameter Selection Guidance ────────────────────────────────")
    print(f"   Objective: screen uncertain parameters for W*_total")
    print(f"   Baseline:  W* = {W_base/G:.4f} kg\n")

    # Use Sobol ST if available, else Morris μ*
    if sobol_results:
        ranked = sorted(sobol_results, key=lambda r: r["ST"], reverse=True)
        metric_name = "ST (Sobol total-order)"
        def _s(r): return r["ST"]
        def _fmt(r): return f"ST={r['ST']:.4f}"
    elif morris_results:
        ranked = morris_results  # already sorted by μ*
        metric_name = "μ* (Morris)"
        def _s(r): return r["mu_star"]
        def _fmt(r): return f"μ*={r['mu_star']:.4f}"
    else:
        print("  No results available.\n")
        return

    keep   = [r for r in ranked if _s(r) >= 0.10]
    middle = [r for r in ranked if 0.02 <= _s(r) < 0.10]
    freeze = [r for r in ranked if _s(r) < 0.02]

    print(f"  Ranked by {metric_name}:\n")
    print(f"  INCLUDE in UQ propagation ({len(keep)} parameters):")
    for i, r in enumerate(keep, 1):
        print(f"    {i:>2}. {r['label']:30s}  {_fmt(r)}  [{r['type']}]")

    if middle:
        print(f"\n  MARGINAL — include if computationally affordable ({len(middle)}):")
        for r in middle:
            print(f"       {r['label']:30s}  {_fmt(r)}  [{r['type']}]")

    if freeze:
        print(f"\n  FREEZE as deterministic ({len(freeze)} parameters, {metric_name} < 0.02):")
        for r in freeze:
            print(f"       {r['label']:30s}  {_fmt(r)}")

    print(f"\n  → For 1-param UMDO: {ranked[0]['label']}")
    if len(ranked) > 1:
        print(f"  → For 2-param UMDO: {ranked[0]['label']}  +  {ranked[1]['label']}")
    if len(ranked) > 2:
        print(f"  → For 3-param UMDO: add  {ranked[2]['label']}")
    print("─────────────────────────────────────────────────────────────────────\n")


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_morris(results: list[dict], W_base: float) -> None:
    """
    Two-panel Morris plot:
    Left:  Ranked bar chart of μ* (importance)
    Right: μ* vs σ scatter (importance vs nonlinearity)
    """
    labels  = [r["label"]   for r in results]
    mu_star = [r["mu_star"] for r in results]
    sigma   = [r["sigma"]   for r in results]
    colors  = [TYPE_COLORS.get(r["type"], "gray") for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, len(labels) * 0.42)))

    # ── Left: ranked bar ──────────────────────────────────────────────────
    y    = np.arange(len(labels))
    xmax = max(mu_star) if max(mu_star) > 0 else 1.0
    ax1.barh(y, mu_star, color=colors, height=0.62, edgecolor="none")
    for yi, v in zip(y, mu_star):
        if v > 0:
            ax1.text(v + xmax * 0.015, yi, f"{v:.4f}",
                     va="center", fontsize=9, color="black")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("μ* = mean|EE| (normalised importance)", fontsize=11)
    ax1.set_title(f"Morris Screening — Importance Ranking\n"
                  f"W*_base = {W_base/G:.3f} kg  "
                  f"(payload={PAYLOAD_KG} kg, R={RANGE_M/1e3:.0f} km)", fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlim(0, xmax * 1.25)
    ax1.axvline(0.02, color="gray", lw=0.8, ls="--", label="Freeze threshold (0.02)")
    ax1.axvline(0.10, color="orange", lw=0.8, ls="--", label="Keep threshold (0.10)")
    ax1.legend(fontsize=9, loc="lower right")

    # ── Right: μ* vs σ scatter ────────────────────────────────────────────
    for r, col in zip(results, colors):
        ax2.scatter(r["mu_star"], r["sigma"], color=col, s=60, zorder=3)
        if r["mu_star"] > 0.02 or r["sigma"] > 0.02:
            ax2.annotate(r["label"], (r["mu_star"], r["sigma"]),
                         fontsize=8, xytext=(4, 4),
                         textcoords="offset points")

    # Draw σ = μ* line (linear model boundary)
    lim = max(max(mu_star), max(sigma)) * 1.1 if max(mu_star) > 0 else 1.0
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8, label="σ = μ* (linear boundary)")
    ax2.set_xlabel("μ* (importance)", fontsize=11)
    ax2.set_ylabel("σ (nonlinearity / interactions)", fontsize=11)
    ax2.set_title("Morris: Importance vs Nonlinearity\n"
                  "(above dashed line → nonlinear or interacting)", fontsize=12)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.legend(fontsize=9)

    # Legend for type colours
    handles = [mpatches.Patch(fc=TYPE_COLORS[t], label=lb)
               for t, lb in [("input", "Input/mission"), ("param", "Parametric"),
                              ("model", "Model-form")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR,
        f"morris_{PAYLOAD_STR}_{N_C}_{RANGE_KM}km_R{R_MORRIS}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()


def plot_sobol(results: list[dict], W_base: float) -> None:
    """
    Sobol bar chart: S1 and ST side-by-side with confidence intervals.
    """
    labels = [r["label"] for r in results]
    S1     = np.array([r["S1"]      for r in results])
    ST     = np.array([r["ST"]      for r in results])
    S1c    = np.array([r["S1_conf"] for r in results])
    STc    = np.array([r["ST_conf"] for r in results])
    colors = [TYPE_COLORS.get(r["type"], "gray") for r in results]

    y    = np.arange(len(labels))
    h    = 0.35
    xmax = max(max(ST), 1.0)

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.45)))
    ax.barh(y + h/2, ST, height=h, color=colors, alpha=0.90, label="ST (total-order)")
    ax.barh(y - h/2, S1, height=h, color=colors, alpha=0.45, hatch="//",
            label="S1 (first-order)")

    # Confidence interval whiskers
    ax.errorbar(ST, y + h/2, xerr=STc, fmt="none", color="black", capsize=3, lw=1)
    ax.errorbar(S1, y - h/2, xerr=S1c, fmt="none", color="black", capsize=3, lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Sobol Sensitivity Index", fontsize=11)
    ax.set_title(f"Sobol Variance-Based Sensitivity — W*_base = {W_base/G:.3f} kg\n"
                 f"payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}  "
                 f"N={N_SOBOL}", fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.15)
    ax.axvline(0.01, color="gray", lw=0.8, ls="--", label="Freeze (ST<0.01)")

    handles = [mpatches.Patch(fc=TYPE_COLORS[t], label=lb)
               for t, lb in [("input", "Input/mission"), ("param", "Parametric"),
                              ("model", "Model-form")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR,
        f"sobol_{PAYLOAD_STR}_{N_C}_{RANGE_KM}km_N{N_SOBOL}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()


def plot_combined(morris_results: list[dict], sobol_results: list[dict],
                  W_base: float) -> None:
    """
    Side-by-side comparison of Morris μ* and Sobol ST on the same parameter list.
    Useful for cross-validation: both methods should agree on the ranking.
    """
    # Merge by attr, keeping Sobol ordering
    sobol_by_attr  = {r["attr"]: r for r in sobol_results}
    morris_by_attr = {r["attr"]: r for r in morris_results}
    attrs_ordered  = [r["attr"] for r in sobol_results]

    labels    = [sobol_by_attr[a]["label"] for a in attrs_ordered]
    ST_vals   = [sobol_by_attr[a]["ST"]    for a in attrs_ordered]
    mu_vals   = [morris_by_attr[a]["mu_star"] if a in morris_by_attr else 0.0
                 for a in attrs_ordered]
    colors    = [TYPE_COLORS.get(sobol_by_attr[a]["type"], "gray") for a in attrs_ordered]

    y    = np.arange(len(labels))
    h    = 0.35
    xmax = max(max(ST_vals), max(mu_vals), 0.01)

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.45)))
    ax.barh(y + h/2, ST_vals, height=h, color=colors, alpha=0.90,
            label="Sobol ST (total-order)")
    ax.barh(y - h/2, mu_vals, height=h, color=colors, alpha=0.45, hatch="//",
            label="Morris μ* (normalised)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Sensitivity Index", fontsize=11)
    ax.set_title(f"Morris vs Sobol Comparison — W*_base = {W_base/G:.3f} kg\n"
                 f"payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.2)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR,
        f"combined_{PAYLOAD_STR}_{N_C}_{RANGE_KM}km.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def write_morris_csv(results: list[dict]) -> None:
    out = os.path.join(RESULTS_DIR,
        f"morris_{PAYLOAD_STR}_{N_C}_{RANGE_KM}km_R{R_MORRIS}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attr", "label", "type", "nominal",
                    "mu_star", "sigma", "sigma_over_mu_star", "n_ee",
                    "decision"])
        for i, r in enumerate(results, 1):
            ratio = r["sigma"] / r["mu_star"] if r["mu_star"] > 1e-10 else 0.0
            if r["mu_star"] < 0.02:
                decision = "FREEZE"
            elif r["mu_star"] < 0.10:
                decision = "marginal"
            else:
                decision = "KEEP"
            w.writerow([i, r["attr"], r["label"], r["type"], r["nominal"],
                        f"{r['mu_star']:.6f}", f"{r['sigma']:.6f}",
                        f"{ratio:.4f}", r["n_ee"], decision])
    print(f"  → {out}")


def write_sobol_csv(results: list[dict]) -> None:
    out = os.path.join(RESULTS_DIR,
        f"sobol_{PAYLOAD_STR}_{N_C}_{RANGE_KM}km_N{N_SOBOL}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attr", "label", "type", "nominal",
                    "S1", "S1_conf", "ST", "ST_conf", "interaction",
                    "decision"])
        for i, r in enumerate(results, 1):
            interaction = r["ST"] - r["S1"]
            if r["ST"] < 0.01:
                decision = "FREEZE"
            elif r["ST"] < 0.05:
                decision = "marginal"
            else:
                decision = "KEEP"
            w.writerow([i, r["attr"], r["label"], r["type"], r["nominal"],
                        f"{r['S1']:.6f}", f"{r['S1_conf']:.6f}",
                        f"{r['ST']:.6f}", f"{r['ST_conf']:.6f}",
                        f"{interaction:.6f}", decision])
    print(f"  → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(SEED)

    print("\n── QBiT UMDO Parameter Importance Screening ─────────────────────────")
    print(f"   Method:   {METHOD.upper()}")
    print(f"   Mission:  payload={PAYLOAD_KG} kg  R={RANGE_M/1e3:.0f} km  n_c={N_C}")
    print(f"   Evaluator: re-optimised W*_total (SLSQP, k={k} parameters)")
    print("─────────────────────────────────────────────────────────────────────\n")

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("Computing baseline W*_total …")
    t0 = time.time()
    W_base = compute_baseline()
    print(f"  W*_base = {W_base/G:.4f} kg  ({time.time()-t0:.1f} s)\n")

    if W_base < 1.0 * G or W_base > 500.0 * G:
        print(f"  [FATAL] Baseline W*={W_base/G:.2f} kg is physically unreasonable.")
        print("  Check your qbit model and constants before proceeding.")
        sys.exit(1)

    morris_results = []
    sobol_results  = []

    # ── Morris ───────────────────────────────────────────────────────────────
    if METHOD in ("morris", "both"):
        morris_results = run_morris(W_base)
        print_morris_table(morris_results, W_base)
        print("Writing Morris outputs …")
        plot_morris(morris_results, W_base)
        write_morris_csv(morris_results)

    # ── Sobol ────────────────────────────────────────────────────────────────
    if METHOD in ("sobol", "both"):
        sobol_results = run_sobol(W_base)
        if sobol_results:
            print_sobol_table(sobol_results, W_base)
            print("Writing Sobol outputs …")
            plot_sobol(sobol_results, W_base)
            write_sobol_csv(sobol_results)

    # ── Combined comparison plot ──────────────────────────────────────────────
    if morris_results and sobol_results:
        plot_combined(morris_results, sobol_results, W_base)

    # ── UMDO guidance ─────────────────────────────────────────────────────────
    umdo_guidance(morris_results, sobol_results, W_base)

    print(f"All outputs written to: {RESULTS_DIR}\n")