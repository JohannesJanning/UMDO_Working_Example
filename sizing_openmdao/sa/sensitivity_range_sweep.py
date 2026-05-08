"""
sensitivity_range_sweep.py

Range-sweep sensitivity viewer inspired by sensitivity_oat.py.

- X axis: one-way range values [5,10,15,20,25,30] km (labels show round-trip: doubled)
- Y axis: MTOM [kg] (optimal W_total / g)
- Solid black line: baseline (no perturbation)
- Fainter lines: MTOM for each parameter perturbed ±10%

Saves PNG and CSV into sizing_openmdao/sa/results_sa_{payload}_{nc}_{min}-{max}

Usage:
    python sensitivity_range_sweep.py
    python sensitivity_range_sweep.py --payload 3 --nc 3

"""
from __future__ import annotations
import os
import sys
import time
import argparse
import importlib
import warnings
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# make local qbit package importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import openmdao.api as om
import qbit.constants as C
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS, DL_MAX, BL_MAX, CL_MAX

# Copy of PARAMS (kept local so script stands alone)
PARAMS = [
    ("RHO_AIR",         1.225,    "rho_air",                "input", False),
    ("T_HOVER",         60.0,     "t_hover",              "input", False),
    ("BETA_QBIT",       0.18,     "beta_QBiT", "param", False),
    ("ETA_HOVER",       0.65,     "eta_hover",                "param", False),
    ("CD0_WING",        0.01,     "CD0_wing",             "param", False),
    ("E_OSWALD",        0.80,     "e_Oswald",             "param", False),
    ("AR_FIXED",        8.0,      "AR",                   "input", False),
    ("SIGMA",           0.13,     "sigma",         "param", False),
    ("CD0_ROTOR",       0.012,    "CD0_rotor",            "param", False),
    ("KAPPA_MAX",       1.15,     "kappa_max",                "param", False),
    ("BATTERY_DENSITY", 158.0,    "rho_bat_Wh_per_kg",        "input", False),
    ("BATTERY_EFF",     0.85,     "eta_bat",                "param", False),
    ("K_MOTOR",         2.506e-4, "k_motor",              "model", False),
    ("K_ESC",           3.594e-4, "k_ESC",                "model", False),
    ("K_ROTOR_A",       0.7484,   "k_rotor_A",            "model", False),
    ("K_ROTOR_B",       0.0403,   "k_rotor_B",            "model", False),
    ("K_WING_A",       -0.0802,   "k_wing_A",             "model", True),
    ("K_WING_B",        2.2854,   "k_wing_B",             "model", False),
]

PERT_FRAC = 0.10
RANGES_KM = [5, 10, 15, 20, 25, 30]

# plotting style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
})


def _restore_all():
    for attr, nominal, *_ in PARAMS:
        setattr(C, attr, nominal)


def _apply_pert(attr, nominal, frac, perturb_abs):
    if perturb_abs:
        val = nominal + frac * abs(nominal)
    else:
        val = nominal * (1.0 + frac)
    setattr(C, attr, val)
    return val


def optimise(payload_kg: float, range_m: float, n_c: int) -> float:
    # reload qbit submodules (excluding constants) so they pick up patched constants
    for name in list(sys.modules.keys()):
        if name.startswith("qbit.") and name != "qbit.constants":
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass
    # rebind builder
    try:
        from qbit.models.qbit_model import build_qbit_model as _build
    except Exception:
        _build = build_qbit_model

    prob = om.Problem(reports=None)
    prob.model = _build(payload_kg, range_m, n_c)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["maxiter"] = 2000

    prob.model.add_design_var("W_total", lower=W_TOTAL_BOUNDS[0], upper=W_TOTAL_BOUNDS[1])
    prob.model.add_design_var("V_inf", lower=V_INF_BOUNDS[0], upper=V_INF_BOUNDS[1])
    prob.model.add_design_var("r", lower=R_BOUNDS[0], upper=R_BOUNDS[1])
    prob.model.add_design_var("J", lower=J_BOUNDS[0], upper=J_BOUNDS[1])
    prob.model.add_design_var("S_w", lower=S_W_BOUNDS[0], upper=S_W_BOUNDS[1])

    prob.model.add_objective("W_total")
    prob.model.add_constraint("weight_residual", equals=0.0)
    prob.model.add_constraint("disk_loading", upper=DL_MAX)
    prob.model.add_constraint("blade_loading", upper=BL_MAX)
    prob.model.add_constraint("cruise_CL", upper=CL_MAX)

    prob.model.set_input_defaults("W_total", val=6.0 * G, units="N")
    prob.setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_driver()
    W_opt = float(prob.get_val("W_total")[0])
    return W_opt / G  # return kg


def run_sweep(payload, n_c, ranges_km):
    min_r = int(min(ranges_km))
    max_r = int(max(ranges_km))
    results_dir = os.path.join(parent_dir, "sa", f"results_sa_{payload}_{n_c}_{min_r}-{max_r}")
    os.makedirs(results_dir, exist_ok=True)

    baseline_vals = []
    perturbed_vals = {p[0]: {"-": [], "+": []} for p in PARAMS}

    total_runs = len(ranges_km) * (1 + 2 * len(PARAMS))
    run_idx = 0
    t0 = time.time()

    for r_km in ranges_km:
        run_idx += 1
        print(f"Running baseline for range {r_km} km ({run_idx}/{total_runs})")
        _restore_all()
        range_m = r_km * 1000.0
        baseline_mt = optimise(payload, range_m, n_c)
        baseline_vals.append(baseline_mt)

        for attr, nominal, *_ in PARAMS:
            # -10%
            run_idx += 1
            _restore_all()
            _apply_pert(attr, nominal, -PERT_FRAC, perturb_abs=False)
            mt_minus = optimise(payload, range_m, n_c)
            perturbed_vals[attr]["-"].append(mt_minus)
            # +10%
            run_idx += 1
            _restore_all()
            _apply_pert(attr, nominal, +PERT_FRAC, perturb_abs=False)
            mt_plus = optimise(payload, range_m, n_c)
            perturbed_vals[attr]["+"].append(mt_plus)

    # plotting
    x = np.array(ranges_km)
    x_labels = [str(int(2 * v)) for v in x]  # round-trip labels in km

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, baseline_vals, marker='o', color='black', lw=2.4, label='Baseline')

    # lighter lines for perturbed params
    for i, (attr, nominal, label, type_tag, perturb_abs) in enumerate(PARAMS):
        color = '#888888'
        y_minus = perturbed_vals[attr]['-']
        y_plus = perturbed_vals[attr]['+']
        # plot minus and plus with low alpha
        ax.plot(x, y_minus, lw=1.2, alpha=0.35, color=color)
        ax.plot(x, y_plus, lw=1.2, alpha=0.35, color=color)

    ax.set_xlabel('Round-trip range [km]')
    ax.set_ylabel('Optimal MTOM [kg]')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(f'Range sweep — payload={payload} kg, n_c={n_c}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    outpng = os.path.join(results_dir, f"range_sweep_payload{payload}_nc{n_c}_{min_r}-{max_r}km.png")
    plt.tight_layout()
    plt.savefig(outpng, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved plot to {outpng}")

    # write CSV: baseline and perturbed
    outcsv = os.path.join(results_dir, f"range_sweep_payload{payload}_nc{n_c}_{min_r}-{max_r}km.csv")
    with open(outcsv, 'w', newline='') as f:
        writer = csv.writer(f)
        # header
        hdr = ['range_km_roundtrip', 'baseline_mt']
        for attr, *_ in PARAMS:
            hdr.append(f'{attr}_minus10_mt')
            hdr.append(f'{attr}_plus10_mt')
        writer.writerow(hdr)
        for i, r_km in enumerate(ranges_km):
            row = [2 * r_km, baseline_vals[i]]
            for attr, *_ in PARAMS:
                row.append(perturbed_vals[attr]['-'][i])
                row.append(perturbed_vals[attr]['+'][i])
            writer.writerow(row)
    print(f"Saved CSV to {outcsv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=float, default=3.0)
    parser.add_argument('--nc', type=int, default=5)
    args = parser.parse_args()
    run_sweep(args.payload, args.nc, RANGES_KM)
