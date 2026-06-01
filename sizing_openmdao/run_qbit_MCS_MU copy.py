"""
run_qbit_MCS_MU.py  –  Robust QBiT sizing with two uncertain parameters.

Uncertain inputs
────────────────
1. T_HOVER   [s]  – hover time per takeoff/landing event
     Shifted lognormal: mean=55 s, std=18 s, lower shift=25 s
     Rationale: operational variability in hover manoeuvres.

2. ETA_HOVER [–]  – rotor hover figure of merit
     Truncated normal: mean=0.65, std=0.04 (~6%), lo=0.50, hi=0.80
     Rationale: aerodynamic figure of merit varies with rotor geometry
     tolerances, blade surface finish, and atmospheric conditions.
     ETA_HOVER couples to W_total through TWO channels:
       (a) P_hover ∝ 1/ETA_HOVER  →  E_req  →  W_battery  (energy path)
       (b) P_hover ∝ 1/ETA_HOVER  →  P_inst →  W_empty    (motor-sizing path)
     This multiplicative interaction with T_HOVER means the joint output
     distribution is non-Gaussian and safety-factor stacking is genuinely
     sub-optimal — UMDO adds value here.

Sampling
────────
Both parameters are sampled jointly with a 2-D Latin Hypercube to preserve
space-filling properties (common random numbers reused across outer
objective evaluations for variance reduction).

Objective
─────────
Minimises the estimated MTOM.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import lognorm, truncnorm
from scipy.stats import qmc
from scipy.optimize import minimize, brentq

try:
    from joblib import Parallel, delayed
    _have_joblib = True
except ImportError:
    _have_joblib = False

import openmdao.api as om
import matplotlib.pyplot as plt
import seaborn as sns

from qbit.models.qbit_model import build_qbit_model
from qbit.components import sizing_comps as sc
from qbit import constants as const
from qbit.constants import (
    G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
    J_BOUNDS, S_W_BOUNDS, BATTERY_EFF, AR_FIXED,
    DL_MAX, BL_MAX, CL_MAX,
    ETA_HOVER as ETA_HOVER_NOMINAL,   # 0.65 – keep nominal for reference
)


# ─────────────────────────────────────────────────────────────────────────────
# ETA_HOVER uncertainty model
# ─────────────────────────────────────────────────────────────────────────────
#
#
ETA_HOVER_MEAN  = 0.65    # –  distribution mean (= nominal)
ETA_HOVER_STD   = 0.05    # –  
ETA_HOVER_LO    = 0.55    # –  lower physical bound
ETA_HOVER_HI    = 0.75    # –  upper physical bound


def _make_eta_hover_dist():
    """Return a frozen truncated-normal for ETA_HOVER."""
    a = (ETA_HOVER_LO - ETA_HOVER_MEAN) / ETA_HOVER_STD
    b = (ETA_HOVER_HI - ETA_HOVER_MEAN) / ETA_HOVER_STD
    return truncnorm(a, b, loc=ETA_HOVER_MEAN, scale=ETA_HOVER_STD)


_ETA_HOVER_DIST = _make_eta_hover_dist()

# 2.5th percentile – fixed value for the deterministic worst-case comparison
eta_hover_deterministic_comparison: float = float(_ETA_HOVER_DIST.ppf(0.025))

print(f"[INFO] ETA_HOVER distribution: mean={ETA_HOVER_MEAN:.3f}, "
      f"std={ETA_HOVER_STD:.3f}")
print(f"[INFO] ETA_HOVER 2.5th pct (deterministic comparison value): "
      f"{eta_hover_deterministic_comparison:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# T_HOVER distribution (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _get_shifted_lognormal(target_mean: float, target_std: float, shift: float):
    mu_prime = target_mean - shift
    v_prime  = target_std ** 2
    s_sq     = math.log(v_prime / mu_prime ** 2 + 1)
    s        = math.sqrt(s_sq)
    scale    = mu_prime / math.sqrt(v_prime / mu_prime ** 2 + 1)
    return lognorm(s, loc=shift, scale=scale)


_T_HOVER_DIST = _get_shifted_lognormal(target_mean=55.0, target_std=18.0, shift=25.0)


# ─────────────────────────────────────────────────────────────────────────────
# Joint 2-D sampler  (T_HOVER, ETA_HOVER)
# ─────────────────────────────────────────────────────────────────────────────

def sample_uncertain_inputs(
    n_samples: int,
    seed: int | None = None,
    method: str = "lhs",
) -> np.ndarray:
    """Draw n_samples joint realisations of (T_HOVER [s], ETA_HOVER [–]).

    Returns array of shape (n_samples, 2).
    Column 0 → T_HOVER   [s]
    Column 1 → ETA_HOVER [–]

    method='lhs' : 2-D Latin Hypercube (recommended)
    method='mcs' : plain Monte Carlo
    """
    if method.lower() in ("lhs", "latin_hypercube", "latin-hypercube"):
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        u = sampler.random(n=n_samples)
        u = np.clip(u, 1e-12, 1 - 1e-12)
    else:
        rng = np.random.default_rng(seed)
        u   = rng.random((n_samples, 2))
        u   = np.clip(u, 1e-12, 1 - 1e-12)

    t_hover   = _T_HOVER_DIST.ppf(u[:, 0])
    eta_hover = _ETA_HOVER_DIST.ppf(u[:, 1])
    return np.column_stack([t_hover, eta_hover])


# ─────────────────────────────────────────────────────────────────────────────
# SizingResult helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingResult:
    W_total: float;  W_battery: float; W_empty: float
    P_hover: float;  P_cruise: float
    V_inf: float;    r: float;         J: float;    S_w: float
    b: float;        chord: float;     E_req: float; converged: bool
    disk_loading: float; blade_loading: float; cruise_CL: float
    weight_residual: float
    DL_MAX: float = None; BL_MAX: float = None; CL_MAX: float = None

    def summary(self) -> str:
        DL_MAX_ = self.DL_MAX if self.DL_MAX is not None else float('nan')
        BL_MAX_ = self.BL_MAX if self.BL_MAX is not None else float('nan')
        CL_MAX_ = self.CL_MAX if self.CL_MAX is not None else float('nan')
        dl_m = DL_MAX_ - self.disk_loading  if not math.isnan(DL_MAX_) else float('nan')
        bl_m = BL_MAX_ - self.blade_loading if not math.isnan(BL_MAX_) else float('nan')
        cl_m = CL_MAX_ - self.cruise_CL     if not math.isnan(CL_MAX_) else float('nan')
        lines = [
            f"  MTOM          : {self.W_total/G:7.3f} kg  ({self.W_total:.1f} N)",
            f"  Battery mass  : {self.W_battery/G:7.3f} kg",
            f"  Empty mass    : {self.W_empty/G:7.3f} kg",
            f"  Cruise speed  : {self.V_inf:7.2f} m/s",
            f"  Rotor radius  : {self.r:7.4f} m",
            f"  Wing area     : {self.S_w:7.4f} m²",
            f"  Wingspan      : {self.b:7.4f} m  (AR={AR_FIXED})",
            f"  Mean chord    : {self.chord:7.4f} m",
            f"  Prop adv. J   : {self.J:5.3f}",
            f"  P_hover       : {self.P_hover:8.1f} W",
            f"  P_cruise      : {self.P_cruise:8.1f} W",
            f"  E_required    : {self.E_req/3600:.3f} Wh",
            f"  Converged     : {self.converged}",
            "--- Constraints & Margins ---",
            f"  Weight Resid. : {self.weight_residual:10.4e} (Goal: 0.0)",
            f"  Disk Loading  : {self.disk_loading:7.2f} / {DL_MAX_} N/m² (Margin: {dl_m:7.2f})",
            f"  Blade Loading : {self.blade_loading:7.4f} / {BL_MAX_}      (Margin: {bl_m:7.4f})",
            f"  Cruise CL     : {self.cruise_CL:7.4f} / {CL_MAX_}      (Margin: {cl_m:7.4f})",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Inner solve  (weight closure for one sample of (t_hover, eta_hover))
# ─────────────────────────────────────────────────────────────────────────────

def inner_solve_for_Wtotal(
    t_hover_sample:   float,
    eta_hover_sample: float,          # replaces e_bat_sample
    payload_kg: float,
    range_m:    float,
    n_c:        int,
    design_vars: Sequence[float],     # [V_inf, r, J, S_w]
    w_initial:  float = 6.0 * G,
) -> dict | float:
    """Weight-closure solve for one realisation of (T_HOVER, ETA_HOVER).

    Patches const.T_HOVER and const.ETA_HOVER at module level so that
    MissionEnergyComp and HoverPowerComp pick up the sampled values.
    Restores originals in the finally block.

    Returns a result dict on success, or np.nan on failure.
    """
    V_inf, r, J, S_w = design_vars

    # ── Problem cache ────────────────────────────────────────────────────────
    key = (float(V_inf), float(r), float(J), float(S_w),
           float(payload_kg), float(range_m), int(n_c))
    if not hasattr(inner_solve_for_Wtotal, '_prob_cache'):
        inner_solve_for_Wtotal._prob_cache = {}

    prob = inner_solve_for_Wtotal._prob_cache.get(key)
    if prob is None:
        prob = om.Problem(reports=None)
        prob.model = build_qbit_model(payload_kg, range_m, n_c)
        prob.model.set_input_defaults('V_inf',  val=float(V_inf),    units='m/s')
        prob.model.set_input_defaults('r',       val=float(r),        units='m')
        prob.model.set_input_defaults('J',       val=float(J))
        prob.model.set_input_defaults('S_w',     val=float(S_w),      units='m**2')
        prob.model.set_input_defaults('W_total', val=float(w_initial), units='N')
        prob.setup()
        inner_solve_for_Wtotal._prob_cache[key] = prob

    # ── Inject / restore uncertain constants ─────────────────────────────────
    def _set_uncertain():
        _orig_T   = getattr(const, 'T_HOVER',   None)
        _orig_ETA = getattr(const, 'ETA_HOVER',  None)
        const.T_HOVER   = float(t_hover_sample)
        const.ETA_HOVER = float(eta_hover_sample)
        return _orig_T, _orig_ETA

    def _restore_uncertain(orig_T, orig_ETA):
        if orig_T is None:
            try: delattr(const, 'T_HOVER')
            except AttributeError: pass
        else:
            const.T_HOVER = orig_T
        if orig_ETA is None:
            try: delattr(const, 'ETA_HOVER')
            except AttributeError: pass
        else:
            const.ETA_HOVER = orig_ETA

    # ── Residual evaluator ───────────────────────────────────────────────────
    def eval_res(W: float) -> float:
        orig_T, orig_ETA = _set_uncertain()
        try:
            prob.set_val('W_total', float(W))
            prob.set_val('V_inf',   float(V_inf))
            prob.set_val('r',       float(r))
            prob.set_val('J',       float(J))
            prob.set_val('S_w',     float(S_w))
            prob.run_model()
            return float(prob.get_val('weight_residual')[0])
        except Exception:
            return float('nan')
        finally:
            _restore_uncertain(orig_T, orig_ETA)

    # ── Extract full result dict at converged W ──────────────────────────────
    def _extract(W_root: float) -> dict:
        orig_T, orig_ETA = _set_uncertain()
        try:
            prob.set_val('W_total', float(W_root))
            prob.run_model()
            return {
                'W_total':          float(prob.get_val('W_total')[0]),
                'W_battery':        float(prob.get_val('W_battery')[0]),
                'W_empty':          float(prob.get_val('W_empty')[0]),
                'P_hover':          float(prob.get_val('P_hover')[0]),
                'P_cruise':         float(prob.get_val('P_cruise')[0]),
                'V_inf':            float(prob.get_val('V_inf')[0]),
                'r':                float(prob.get_val('r')[0]),
                'J':                float(prob.get_val('J')[0]),
                'S_w':              float(prob.get_val('S_w')[0]),
                'E_req':            float(prob.get_val('E_req')[0]),
                'disk_loading':     float(prob.get_val('disk_loading')[0]),
                'blade_loading':    float(prob.get_val('blade_loading')[0]),
                'cruise_CL':        float(prob.get_val('cruise_CL')[0]),
                'weight_residual':  float(prob.get_val('weight_residual')[0]),
                # record sampled uncertain inputs for diagnostics
                't_hover_sample':   float(t_hover_sample),
                'eta_hover_sample': float(eta_hover_sample),
            }
        finally:
            _restore_uncertain(orig_T, orig_ETA)

    # ── Brent root-find on W_total ───────────────────────────────────────────
    wl, wh = float(W_TOTAL_BOUNDS[0]), float(W_TOTAL_BOUNDS[1])
    try:
        Ns   = 12
        xs   = np.linspace(wl, wh, Ns)
        vals = [eval_res(x) for x in xs]

        bracket_found = False
        for i in range(len(xs) - 1):
            fa, fb = vals[i], vals[i + 1]
            if math.isnan(fa) or math.isnan(fb):
                continue
            if fa * fb <= 0:
                wl, wh = xs[i], xs[i + 1]
                bracket_found = True
                break

        if not bracket_found:
            return float('nan')

        W_root = brentq(eval_res, wl, wh, xtol=1e-3, rtol=1e-4, maxiter=60)
        return _extract(W_root)

    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Robust optimiser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobustOptimizer:
    payload_kg: float
    range_m:    float
    n_c:        int   = 1
    n_mc:       int   = 50
    seed:       int | None = 123
    mc_samples: np.ndarray | None = None   # shape (n_mc, 2) once generated
    sampling_method: str  = "lhs"
    cl_margin:       float = 0.0000
    cl_reliability:  float = 0.975
    n_jobs:          int   = 1
    maxiter:         int   = 200
    _obj_calls:       int        = field(default=0, repr=False)
    eval_times:       list       = field(default=None, repr=False)
    inner_calls:      int        = field(default=0, repr=False)
    inner_time_total: float      = field(default=0.0, repr=False)

    def objective(self, x: Sequence[float]) -> float:
        """Minimise estimated MTOM."""
        V_inf, r, J, S_w = x
        t0    = time.time()
        stats = self._mc_stats(x)
        t1    = time.time()

        if self.eval_times is None:
            self.eval_times = []
        self.eval_times.append(t1 - t0)
        self._obj_calls += 1

        bar_len   = 20
        avg       = float(np.mean(self.eval_times)) if self.eval_times else 0.0
        remaining = max(0, self.maxiter - self._obj_calls)
        eta_s     = avg * remaining
        h, m, s   = int(eta_s // 3600), int((eta_s % 3600) // 60), int(eta_s % 60)
        frac      = min(1.0, self._obj_calls / self.maxiter) if self.maxiter > 0 else 0.0
        bar       = "[" + "#" * int(frac * bar_len) + " " * (bar_len - int(frac * bar_len)) + "]"

        if stats is None:
            print(f"\r{bar} [{self._obj_calls}/{self.maxiter}] ETA {h}h{m}m{s}s → FAILED", end="")
            return 1e6

        meanW   = stats['meanW']
        stdW    = stats['stdW']
        p97_5_W = meanW + 1.96 * stdW

        print(f"\r{bar} [{self._obj_calls}/{self.maxiter}] ETA {h}h{m}m{s}s "
              f"→ V={V_inf:.2f}, r={r:.3f}, S_w={S_w:.3f} "
              f"mean={meanW/G:.2f}kg  p97.5={p97_5_W/G:.2f}kg",
              end="", flush=True)

        self._last_results = stats['results']
        return float(p97_5_W)

    def _mc_stats(self, x: Sequence[float]) -> dict | None:
        V_inf, r, J, S_w = x

        if self.mc_samples is None:
            samples = sample_uncertain_inputs(self.n_mc, seed=self.seed,
                                             method=self.sampling_method)
        else:
            samples = self.mc_samples

        t0 = time.time()

        def _call(row):
            t_h, eta_h = float(row[0]), float(row[1])
            return inner_solve_for_Wtotal(
                t_h, eta_h, self.payload_kg, self.range_m, self.n_c,
                design_vars=(V_inf, r, J, S_w),
            )

        if self.n_jobs != 1 and _have_joblib:
            try:
                res_list = Parallel(n_jobs=self.n_jobs, prefer='processes')(
                    delayed(_call)(samples[i]) for i in range(len(samples))
                )
            except Exception:
                res_list = [_call(samples[i]) for i in range(len(samples))]
        else:
            res_list = [_call(samples[i]) for i in range(len(samples))]

        t1 = time.time()
        self.inner_calls      += len(samples)
        self.inner_time_total += (t1 - t0)

        results, W_vals = [], []
        for res in res_list:
            if res is None or (isinstance(res, float) and math.isnan(res)):
                results.append(None)
                W_vals.append(float('nan'))
            elif isinstance(res, dict):
                results.append(res)
                W_vals.append(res['W_total'])
            else:
                results.append(None)
                W_vals.append(float(res))

        W_arr = np.array(W_vals)
        if np.isnan(W_arr).any():
            return None

        meanW        = float(np.mean(W_arr))
        stdW         = float(np.std(W_arr, ddof=0))
        p97_5_W      = meanW + 1.96 * stdW
        p97_5_sample = float(np.percentile(W_arr, 97.5))

        keys = ['W_battery', 'W_empty', 'P_hover', 'P_cruise', 'V_inf', 'r',
                'J', 'S_w', 'E_req', 'disk_loading', 'blade_loading',
                'cruise_CL', 'weight_residual']
        mean_res = {}
        for k in keys:
            vals = np.array([p[k] for p in results if isinstance(p, dict)])
            mean_res[k] = float(np.mean(vals)) if vals.size > 0 else float('nan')

        return {
            'meanW':         meanW,
            'stdW':          stdW,
            'p97_5W':        p97_5_W,
            'p97_5_sample':  p97_5_sample,
            'mean_res':      mean_res,
            'results':       results,
            'W_samples':     W_arr,
        }

    #def constr_cruise_CL(self, x: Sequence[float]) -> float:
     #           """CL constraint: P(CL ≤ CL_MAX) ≥ cl_reliability (default 97.5%)"""
      #          stats = self._mc_stats(x)
       #         if stats is None: 
        #            return -1e6
         #       
          #      cl_arr = np.array([p['cruise_CL'] for p in stats['results'] 
           #                     if isinstance(p, dict)])
            #    if cl_arr.size == 0: 
             #       return -1e6
              #  # Empirical percentile (no normality assumption)
               # cl_percentile = np.percentile(cl_arr, 100 * self.cl_reliability)
                #return float(CL_MAX - cl_percentile - self.cl_margin)


    def run(self, x0: Sequence[float] | None = None, method: str = 'SLSQP'):
        bounds = [V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS]
        if x0 is None:
            x0 = [np.mean(b) for b in bounds]

        self.mc_samples = sample_uncertain_inputs(
            self.n_mc, seed=self.seed, method=self.sampling_method
        )
        print(f"Starting robust optimisation with {self.n_mc} joint (T_HOVER, ETA_HOVER) samples.")
        print(f"  T_HOVER  : shifted-lognormal  mean=55 s, std=18 s")
        print(f"  ETA_HOVER: truncated-normal   mean={ETA_HOVER_MEAN:.3f}, "
              f"std={ETA_HOVER_STD:.3f}, lo={ETA_HOVER_LO:.2f}, hi={ETA_HOVER_HI:.2f}")

        def constr_cruise_CL(x):
            stats = self._mc_stats(x)
            if stats is None: return -1e6
            return float(CL_MAX - stats['mean_res']['cruise_CL'] - self.cl_margin)


        def constr_disk_loading(x):
            stats = self._mc_stats(x)
            if stats is None: return -1e6
            return float(DL_MAX - stats['mean_res']['disk_loading'])

        def constr_blade_loading(x):
            stats = self._mc_stats(x)
            if stats is None: return -1e6
            return float(BL_MAX - stats['mean_res']['blade_loading'])

        cons = [
            {'type': 'ineq', 'fun': constr_cruise_CL},
            {'type': 'ineq', 'fun': constr_disk_loading},
            {'type': 'ineq', 'fun': constr_blade_loading},
        ]

        self._obj_calls       = 0
        self.eval_times       = []
        self.inner_calls      = 0
        self.inner_time_total = 0.0

        t_start = time.time()
        res = minimize(self.objective, x0, method=method, bounds=bounds,
                       constraints=cons,
                       options={'maxiter': self.maxiter, 'ftol': 1e-6, 'disp': True})
        t_end = time.time()

        total_t   = t_end - t_start
        avg_obj   = float(np.mean(self.eval_times)) if self.eval_times else float('nan')
        avg_inner = (self.inner_time_total / self.inner_calls
                     if self.inner_calls > 0 else float('nan'))

        print('\n--- Optimisation Diagnostics ---')
        print(f' Method           : {method}')
        print(f' Success          : {res.success}')
        print(f' Message          : {res.message}')
        print(f' Objective value  : {res.fun:.4f}  ({res.fun/G:.4f} kg)')
        print(f' x*               : V_inf={res.x[0]:.2f}, r={res.x[1]:.4f}, '
              f'J={res.x[2]:.3f}, S_w={res.x[3]:.4f}')
        print(f' Total time       : {total_t:.1f} s')
        print(f' Objective evals  : {self._obj_calls} (avg {avg_obj:.3f} s/eval)')
        print(f' Inner solves     : {self.inner_calls} (avg {avg_inner:.4f} s/solve)')
        print(f' MC per eval      : {self.n_mc}  (2 uncertain params, 2-D LHS)')
        print(f' Seed             : {self.seed}')
        print('--------------------------------\n')
        return res


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helper
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(
    x_opt:      Sequence[float],
    payload_kg: float,
    range_m:    float,
    n_c:        int,
    n_mc_large: int  = 2000,
    seed:       int  = 123,
    out_prefix: str  = 'sizing_openmdao/robust',
):
    """Evaluate the optimised design on a large MC sample and plot results."""
    samples = sample_uncertain_inputs(n_mc_large, seed=seed)
    results = []
    for row in samples:
        r = inner_solve_for_Wtotal(
            float(row[0]), float(row[1]),
            payload_kg, range_m, n_c,
            design_vars=tuple(x_opt),
        )
        results.append(r if isinstance(r, dict) else None)

    valid = [p for p in results if isinstance(p, dict)]
    if not valid:
        print('No valid MC samples in post-processing.')
        return

    Wtot  = np.array([p['W_total'] for p in valid])
    meanW = float(np.mean(Wtot))
    stdW  = float(np.std(Wtot, ddof=0))
    p_lo, p_hi = np.percentile(Wtot / G, [2.5, 97.5])

    keys = ['W_battery', 'W_empty', 'P_hover', 'P_cruise', 'V_inf', 'r',
            'J', 'S_w', 'E_req', 'disk_loading', 'blade_loading',
            'cruise_CL', 'weight_residual']
    mrd = {k: float(np.mean([p[k] for p in valid])) for k in keys}

    b     = float(np.sqrt(AR_FIXED * mrd['S_w']))
    chord = mrd['S_w'] / b
    sr = SizingResult(
        W_total=meanW, W_battery=mrd['W_battery'], W_empty=mrd['W_empty'],
        P_hover=mrd['P_hover'], P_cruise=mrd['P_cruise'],
        V_inf=mrd['V_inf'], r=mrd['r'], J=mrd['J'], S_w=mrd['S_w'],
        b=b, chord=chord, E_req=mrd['E_req'], converged=True,
        disk_loading=mrd['disk_loading'], blade_loading=mrd['blade_loading'],
        cruise_CL=mrd['cruise_CL'], weight_residual=mrd['weight_residual'],
        DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX,
    )
    print('\n--- Robust Design Summary (optimised design, large MC) ---')
    print(sr.summary())
    print(f'\nMTOM (kg)  mean={meanW/G:.3f}  std={stdW/G:.3f}  '
          f'95% PI=[{p_lo:.3f}, {p_hi:.3f}]')
    print(f'\nUncertainty parameter ranges in this MC:')
    t_arr   = np.array([p['t_hover_sample']   for p in valid])
    eta_arr = np.array([p['eta_hover_sample']  for p in valid])
    print(f'  T_HOVER [s]    : [{t_arr.min():.1f}, {t_arr.max():.1f}]  '
          f'mean={t_arr.mean():.1f}')
    print(f'  ETA_HOVER [–]  : [{eta_arr.min():.3f}, {eta_arr.max():.3f}]  '
          f'mean={eta_arr.mean():.3f}')

    try:
        import os; os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

        mtom_kg = Wtot / G
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)

        ax = axes[0]
        sns.histplot(mtom_kg, bins=60, stat='density', color='C0', kde=True, ax=ax)
        ax.axvline(np.mean(mtom_kg), color='k',  ls='--', lw=1.5,
                   label=f'Mean {np.mean(mtom_kg):.3f} kg')
        ax.axvline(p_lo,             color='r',  ls=':',  lw=1.5,
                   label=f'2.5%  {p_lo:.3f} kg')
        ax.axvline(p_hi,             color='r',  ls=':',  lw=1.5,
                   label=f'97.5% {p_hi:.3f} kg')
        ax.set_xlabel('MTOM (kg)');  ax.set_ylabel('Density')
        ax.set_title('MTOM distribution');  ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        sc_ = ax.scatter(t_arr, mtom_kg, c=eta_arr, cmap='RdYlGn', s=8, alpha=0.6,
                         vmin=ETA_HOVER_LO, vmax=ETA_HOVER_HI)
        plt.colorbar(sc_, ax=ax, label='η_hover [–]')
        ax.set_xlabel('T_HOVER (s)');  ax.set_ylabel('MTOM (kg)')
        ax.set_title('T_HOVER vs MTOM\n(colour = η_hover)');  ax.grid(alpha=0.3)

        ax = axes[2]
        sc2 = ax.scatter(eta_arr, mtom_kg, c=t_arr, cmap='plasma', s=8, alpha=0.6)
        plt.colorbar(sc2, ax=ax, label='T_HOVER (s)')
        ax.set_xlabel('η_hover [–]');  ax.set_ylabel('MTOM (kg)')
        ax.set_title('η_hover vs MTOM\n(colour = T_HOVER)');  ax.grid(alpha=0.3)

        plt.suptitle('Robust QBiT sizing  –  joint (T_HOVER, ETA_HOVER) uncertainty',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(f'{out_prefix}_mtom_{ext}.{ext}', dpi=180,
                        bbox_inches='tight')
            print(f'Saved {out_prefix}_mtom_{ext}.{ext}')
        plt.close()

    except Exception as e:
        print(f'Plotting failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = RobustOptimizer(
        payload_kg = 3.0,
        range_m    = 15_000.0,
        n_c        = 2,
        n_mc       = 100,
        seed       = 123,
    )
    x0  = [33.0, 0.22, 1.3, 0.2]
    res = opt.run(x0=x0, method='SLSQP')
    print('\nOptimisation completed.')
