"""
run_qbit_robust.py - Robust sizing optimization for QBiT with uncertain hover time.

Minimize U = 0.5 * mean(W_total) + 0.5 * std(W_total) where the uncertainty in
`T_HOVER` is propagated via Monte Carlo sampling. For each Monte Carlo sample
an inner OpenMDAO solve finds the required `W_total` (weight closure) for the
given design variables.

Notes:
- This implements a nested (outer) optimizer over design variables
  [V_inf, r, J, S_w] and (inner) solves that compute `W_total` for sampled
  `T_HOVER` values. The implementation is intentionally simple and
  sequential; Monte Carlo and inner solves are expensive.
"""
from __future__ import annotations
import math
import time
import sys
import shutil
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import lognorm
from scipy.stats import qmc
from scipy.optimize import minimize, brentq

try:
    from joblib import Parallel, delayed  # optional, for parallel inner solves
    _have_joblib = True
except Exception:
    _have_joblib = False

import openmdao.api as om

from qbit.models.qbit_model import build_qbit_model
from qbit import components as q_components
from qbit.components import sizing_comps as sc
from qbit.constants import (G, W_TOTAL_BOUNDS, V_INF_BOUNDS, R_BOUNDS,
                             J_BOUNDS, S_W_BOUNDS, BATTERY_EFF, AR_FIXED,
                             DL_MAX, BL_MAX, CL_MAX)
import matplotlib.pyplot as plt


@dataclass
class SizingResult:
    W_total: float; W_battery: float; W_empty: float
    P_hover: float; P_cruise: float
    V_inf: float; r: float; J: float; S_w: float
    b: float; chord: float; E_req: float; converged: bool
    disk_loading: float; blade_loading: float; cruise_CL: float; weight_residual: float
    DL_MAX: float = None; BL_MAX: float = None; CL_MAX: float = None

    def summary(self) -> str:
        DL_MAX = self.DL_MAX if self.DL_MAX is not None else float('nan')
        BL_MAX = self.BL_MAX if self.BL_MAX is not None else float('nan')
        CL_MAX = self.CL_MAX if self.CL_MAX is not None else float('nan')
        dl_margin = DL_MAX - self.disk_loading if not math.isnan(DL_MAX) else float('nan')
        bl_margin = BL_MAX - self.blade_loading if not math.isnan(BL_MAX) else float('nan')
        cl_margin = CL_MAX - self.cruise_CL if not math.isnan(CL_MAX) else float('nan')
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
            f"  Disk Loading  : {self.disk_loading:7.2f} / {DL_MAX} N/m² (Margin: {dl_margin:7.2f})",
            f"  Blade Loading : {self.blade_loading:7.4f} / {BL_MAX}      (Margin: {bl_margin:7.4f})",
            f"  Cruise CL     : {self.cruise_CL:7.4f} / {CL_MAX}      (Margin: {cl_margin:7.4f})",
        ]
        return "\n".join(lines)


def get_shifted_lognormal_dist(target_mean: float, target_std: float, shift: float):
    mu_prime = target_mean - shift
    v_prime = target_std**2
    s_sq = math.log(v_prime / mu_prime**2 + 1)
    s = math.sqrt(s_sq)
    scale = mu_prime / math.sqrt(v_prime / mu_prime**2 + 1)
    return lognorm(s, loc=shift, scale=scale)


def sample_t_hover(n_samples: int, mean: float = 55.0, std: float = 18.0,
                   shift: float = 25.0, seed: int | None = None,
                   method: str = "lhs") -> np.ndarray:
    """Generate samples for T_HOVER using the chosen sampling method.

    method: 'lhs' for Latin Hypercube Sampling (recommended) or 'mcs'
            for plain Monte Carlo sampling.
    Returns an array of shape (n_samples,) in the physical units of t_hover.
    """
    dist = get_shifted_lognormal_dist(mean, std, shift)
    if method.lower() in ("lhs", "latin", "latin_hypercube", "latin-hypercube"):
        # Latin Hypercube in unit [0,1], then map via distribution PPF
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        u = sampler.random(n=n_samples).ravel()
        # avoid exact 0/1 which can map to -inf/inf for some dists
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return dist.ppf(u)
    else:
        rng = np.random.default_rng(seed)
        return dist.rvs(size=n_samples, random_state=rng)


def inner_solve_for_Wtotal(t_hover_sample: float, payload_kg: float, range_m: float, n_c: int,
                           design_vars: Sequence[float], w_initial: float = 6.0 * G) -> float:
    """For a single sampled `t_hover`, build the OpenMDAO problem with that
    hover time and the given fixed design variables and solve only for
    `W_total` (the inner weight-closure solve). Returns found W_total (N) or
    np.nan on failure.
    design_vars: [V_inf, r, J, S_w]
    """
    # We'll solve weight_residual(W_total) == 0 by root-finding on W_total.
    # Cache Problems per design point to avoid repeated setup overhead.
    V_inf, r, J, S_w = design_vars

    key = (float(V_inf), float(r), float(J), float(S_w), float(payload_kg), float(range_m), int(n_c))
    if not hasattr(inner_solve_for_Wtotal, '_prob_cache'):
        inner_solve_for_Wtotal._prob_cache = {}

    prob = inner_solve_for_Wtotal._prob_cache.get(key)
    if prob is None:
        prob = om.Problem(reports=None)
        prob.model = build_qbit_model(payload_kg, range_m, n_c)
        # Register W_total as a model input/output (we will set it directly)
        prob.model.add_output = getattr(prob.model, 'add_output', None)
        # Set fixed design variable defaults
        prob.model.set_input_defaults('V_inf', val=float(V_inf), units='m/s')
        prob.model.set_input_defaults('r',     val=float(r),     units='m')
        prob.model.set_input_defaults('J',     val=float(J))
        prob.model.set_input_defaults('S_w',   val=float(S_w),   units='m**2')
        prob.model.set_input_defaults('W_total', val=float(w_initial), units='N')
        prob.setup()
        # Store in cache
        inner_solve_for_Wtotal._prob_cache[key] = prob

    # Define residual evaluator
    def eval_res(W: float) -> float:
        # set hover time and W_total, run model (not driver)
        orig_T = getattr(sc, 'T_HOVER', None)
        sc.T_HOVER = float(t_hover_sample)
        try:
            prob.set_val('W_total', float(W))
            prob.set_val('V_inf', float(V_inf))
            prob.set_val('r', float(r))
            prob.set_val('J', float(J))
            prob.set_val('S_w', float(S_w))
            # run_model executes the model without running a driver
            prob.run_model()
            rr = float(prob.get_val('weight_residual')[0])
            return rr
        except Exception:
            return float('nan')
        finally:
            if orig_T is None:
                delattr(sc, 'T_HOVER')
            else:
                sc.T_HOVER = orig_T

    # Try to bracket root within W_TOTAL_BOUNDS
    wl, wh = float(W_TOTAL_BOUNDS[0]), float(W_TOTAL_BOUNDS[1])
    try:
        rl = eval_res(wl)
        rh = eval_res(wh)
        if math.isnan(rl) or math.isnan(rh):
            return float('nan')
        if rl == 0.0:
            return wl
        if rh == 0.0:
            return wh
        if rl * rh > 0:
            # no sign change; attempt scanning for bracket
            Ns = 9
            xs = np.linspace(wl, wh, Ns)
            vals = [eval_res(x) for x in xs]
            for i in range(len(xs)-1):
                a, b = xs[i], xs[i+1]
                fa, fb = vals[i], vals[i+1]
                if not (math.isnan(fa) or math.isnan(fb)) and fa * fb <= 0:
                    wl, wh, rl, rh = a, b, fa, fb
                    break
            else:
                return float('nan')

        # root find
        W_root = brentq(eval_res, wl, wh, xtol=1e-3, rtol=1e-4, maxiter=100)
        # extract full model outputs at root
        orig_T = getattr(sc, 'T_HOVER', None)
        sc.T_HOVER = float(t_hover_sample)
        try:
            prob.set_val('W_total', float(W_root))
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
        finally:
            if orig_T is None:
                delattr(sc, 'T_HOVER')
            else:
                sc.T_HOVER = orig_T
    except Exception:
        # fallback: try running the original inner optimizer as a last resort
        try:
            orig_T = getattr(sc, 'T_HOVER', None)
            sc.T_HOVER = float(t_hover_sample)
            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['tol'] = 1e-6
            prob.driver.options['maxiter'] = 500
            prob.run_driver()
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
            return float('nan')


@dataclass
class RobustOptimizer:
    payload_kg: float
    range_m: float
    n_c: int = 1
    n_mc: int = 50
    mean_t: float = 55.0
    std_t: float = 18.0
    shift_t: float = 25.0
    seed: int | None = 123
    mc_samples: np.ndarray | None = None
    sampling_method: str = "lhs"
    cl_margin: float = 0.0001  # small margin to ensure we don't optimize right up to the CL_MAX constraint
    # diagnostics (filled in `run`)
    _obj_calls: int = 0
    eval_times: list | None = None
    inner_calls: int = 0
    inner_time_total: float = 0.0
    n_jobs: int = 10
    maxiter: int = 200

    def objective(self, x: Sequence[float]) -> float:
        """Outer objective to minimize: U = 0.5*mean(W) + 0.5*std(W)
        where W are W_total values computed by inner solves across MC samples.
        x is [V_inf, r, J, S_w]
        """
        V_inf, r, J, S_w = x
        t0 = time.time()
        stats = self._mc_stats(x)
        t1 = time.time()
        # record timing
        if self.eval_times is None:
            self.eval_times = []
        self.eval_times.append(t1 - t0)
        self._obj_calls = getattr(self, '_obj_calls', 0) + 1
        # best-effort ETA: estimate remaining iterations as (maxiter - obj_calls)
        try:
            avg = float(np.mean(self.eval_times))
        except Exception:
            avg = float('nan')
        remaining = max(0, int(self.maxiter - self._obj_calls))
        eta_s = avg * remaining if not math.isnan(avg) else float('nan')

        # Build a compact progress bar and single-line status update.
        bar_len = 24
        if self.maxiter > 0:
            frac = min(1.0, float(self._obj_calls) / float(self.maxiter))
        else:
            frac = 0.0
        filled = int(frac * bar_len)
        bar = "[" + "#" * filled + " " * (bar_len - filled) + "]"

        # Format ETA
        try:
            hrs = int(eta_s // 3600)
            mins = int((eta_s % 3600) // 60)
            secs = int(eta_s % 60)
            eta_str = f"{hrs}h{mins}m{secs}s"
        except Exception:
            eta_str = "--"

        meanW = None
        stdW = None
        U = float('nan')
        if stats is None:
            status = "FAILED"
            U = 1e6
        else:
            meanW = stats['meanW']
            stdW = stats['stdW']
            K = 0
            U = (1-K) * meanW + K * stdW
            status = "OK"

        # Build status line and pad to terminal width to avoid leftover chars.
        term_w = shutil.get_terminal_size((120, 20)).columns
        if meanW is None:
            body = f"{bar} [{self._obj_calls}/{self.maxiter}] ETA {eta_str} → V={V_inf:.3f}, r={r:.4f}, J={J:.3f}, S_w={S_w:.4f} {status} U={U:.1f}"
        else:
            body = f"{bar} [{self._obj_calls}/{self.maxiter}] ETA {eta_str} → V={V_inf:.3f}, r={r:.4f}, J={J:.3f}, S_w={S_w:.4f} mean={meanW:.1f}N std={stdW:.1f}N U={U:.1f}"
        # ensure line not longer than terminal, truncate if needed
        if len(body) > term_w - 1:
            body = body[:term_w-4] + "..."

        # Print single-line status (carriage return) and keep it on the same line.
        sys.stdout.write('\r' + body.ljust(term_w - 1))
        sys.stdout.flush()
        self._last_results = stats['results'] if stats is not None else None
        return U

    def _mc_stats(self, x: Sequence[float]):
        """Evaluate MC samples for design x and return aggregated stats.
        Returns dict with keys: meanW, stdW, mean_metrics_dict, results (list), or
        None on failure.
        """
        V_inf, r, J, S_w = x
        if self.mc_samples is None:
            samples = sample_t_hover(self.n_mc, self.mean_t, self.std_t, self.shift_t,
                                     seed=self.seed, method=self.sampling_method)
        else:
            samples = self.mc_samples
        results = []
        W_vals = []
        t_inner0 = time.time()

        # Parallelize inner solves when requested and joblib is available.
        if getattr(self, 'n_jobs', 1) is not None and int(self.n_jobs) != 1 and _have_joblib:
            try:
                workers = int(self.n_jobs)
                # wrapper to keep signature picklable for joblib
                def _call_inner(t_val):
                    return inner_solve_for_Wtotal(t_val, self.payload_kg, self.range_m, self.n_c,
                                                  design_vars=(V_inf, r, J, S_w))

                res_list = Parallel(n_jobs=workers, prefer='processes')(delayed(_call_inner)(t) for t in samples)
            except Exception:
                # fallback to sequential if parallelization fails
                res_list = [inner_solve_for_Wtotal(t, self.payload_kg, self.range_m, self.n_c,
                                                   design_vars=(V_inf, r, J, S_w)) for t in samples]
        else:
            res_list = [inner_solve_for_Wtotal(t, self.payload_kg, self.range_m, self.n_c,
                                               design_vars=(V_inf, r, J, S_w)) for t in samples]

        t_inner1 = time.time()
        # update counters
        self.inner_calls = getattr(self, 'inner_calls', 0) + len(samples)
        self.inner_time_total = getattr(self, 'inner_time_total', 0.0) + (t_inner1 - t_inner0)

        # collect results
        for res in res_list:
            if res is None or (isinstance(res, float) and math.isnan(res)):
                results.append(None)
                W_vals.append(float('nan'))
            else:
                if isinstance(res, dict):
                    results.append(res)
                    W_vals.append(res.get('W_total', float('nan')))
                else:
                    results.append(None)
                    W_vals.append(float(res))
        W_arr = np.array(W_vals)
        if np.isnan(W_arr).any():
            return None
        meanW = float(np.mean(W_arr))
        stdW = float(np.std(W_arr, ddof=0))
        # average other fields
        keys = ['W_battery','W_empty','P_hover','P_cruise','V_inf','r','J','S_w','E_req','disk_loading','blade_loading','cruise_CL','weight_residual']
        mean_res = {}
        for k in keys:
            vals = np.array([p[k] for p in results if isinstance(p, dict)])
            mean_res[k] = float(np.mean(vals)) if vals.size>0 else float('nan')
        return {'meanW': meanW, 'stdW': stdW, 'mean_res': mean_res, 'results': results}

    def run(self, x0: Sequence[float] | None = None, method: str = 'SLSQP'):
        # bounds
        bounds = [V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS]
        if x0 is None:
            x0 = [np.mean(b) for b in bounds]
        # Pre-generate common MC samples (CRN) for all objective evaluations
        self.mc_samples = sample_t_hover(self.n_mc, self.mean_t, self.std_t, self.shift_t,
                         seed=self.seed, method=self.sampling_method)
        print("Starting robust optimization (using common random numbers).")
        # Build constraints enforcing mean metrics across MC samples
        def constr_cruise_CL(x):
            stats = self._mc_stats(x)
            if stats is None:
                return -1e6
            # Reliability-based (chance) constraint: enforce that the
            # (one-sided) 95th percentile of cruise CL is below CL_MAX.
            # Use mean + 1.645*std as the approximate 95th percentile.
            # Safely compute std from the per-sample results.
            # For 95% reliability, use 1.645
            # For 99% reliability, use 2.326
            # For 99.9% reliability, use 3.09
            cl_samples = np.array([p['cruise_CL'] for p in stats['results'] if isinstance(p, dict)])
            if cl_samples.size == 0:
                return -1e6
            mean_cl = float(stats['mean_res']['cruise_CL'])
            std_cl = float(np.std(cl_samples, ddof=0))
            # enforce: CL_MAX - (mean + 1.645*std + margin) >= 0
            return float(CL_MAX - (mean_cl + 2.326 * std_cl) - self.cl_margin)

        def constr_disk_loading(x):
            stats = self._mc_stats(x)
            if stats is None:
                return -1e6
            return float(DL_MAX - stats['mean_res']['disk_loading'])

        def constr_blade_loading(x):
            stats = self._mc_stats(x)
            if stats is None:
                return -1e6
            return float(BL_MAX - stats['mean_res']['blade_loading'])

        cons = [
            {'type': 'ineq', 'fun': constr_cruise_CL},
            {'type': 'ineq', 'fun': constr_disk_loading},
            {'type': 'ineq', 'fun': constr_blade_loading},
        ]

        # Use SLSQP with bounds and constraints to enforce limits during optimization
        t_start = time.time()
        # reset diagnostics
        self._obj_calls = 0
        self.eval_times = []
        self.inner_calls = 0
        self.inner_time_total = 0.0
        # pass maxiter into optimizer and record it for ETA estimates
        opt_options = {'maxiter': self.maxiter, 'ftol': 1e-6, 'disp': True}
        res = minimize(self.objective, x0, method=method, bounds=bounds, constraints=cons,
                   options=opt_options)
        t_end = time.time()
        # finish progress line
        try:
            term_w = shutil.get_terminal_size((120, 20)).columns
            sys.stdout.write('\r' + ' ' * (term_w - 1) + '\r')
        except Exception:
            print()

        # Print diagnostic summary useful for scientific reporting
        total_opt_time = t_end - t_start
        n_obj = getattr(self, '_obj_calls', 0)
        avg_obj_time = float(np.mean(self.eval_times)) if self.eval_times else float('nan')
        total_inner = int(getattr(self, 'inner_calls', 0))
        avg_inner_time = float(self.inner_time_total / total_inner) if total_inner > 0 else float('nan')
        print('\n--- Optimization Diagnostics ---')
        print(f' Method         : {method}')
        print(f' Success        : {getattr(res, "success", None)}')
        print(f' Message        : {getattr(res, "message", "")}')
        print(f' Objective vals : {getattr(res, "fun", None):.6f}')
        print(f' x              : {getattr(res, "x", None)}')
        print(f' Total time     : {total_opt_time:.2f} s')
        print(f' Objective evals: {n_obj} (avg {avg_obj_time:.3f} s/eval)')
        print(f' Inner solves   : {total_inner} (avg {avg_inner_time:.3f} s/solve)')
        print(f' MC per eval    : {self.n_mc}')
        print(f' Seed           : {self.seed}')
        print('---------------------------------\n')

        return res


if __name__ == '__main__':
    # Quick test run with reduced Monte Carlo for speed
    opt = RobustOptimizer(payload_kg=3.0, range_m=15_000.0, n_c=2, n_mc=100, seed=123)
    x0 = [33.0, 0.22, 1.3, 0.2]
    res = opt.run(x0=x0, method='SLSQP')
    print('\nOptimization finished:')
    print(res)

    # Post-process: evaluate optimized design with larger MC and print styled summary
    x_opt = res.x
    opt_large = RobustOptimizer(payload_kg=opt.payload_kg, range_m=opt.range_m, n_c=opt.n_c,
                                n_mc=2000, mean_t=opt.mean_t, std_t=opt.std_t, shift_t=opt.shift_t, seed=opt.seed,
                                sampling_method=opt.sampling_method)
    samples = opt_large.mc_samples = sample_t_hover(opt_large.n_mc, opt_large.mean_t, opt_large.std_t, opt_large.shift_t,
                                                    seed=opt_large.seed, method=opt_large.sampling_method)
    per_results = []
    for t in samples:
        r = inner_solve_for_Wtotal(t, opt_large.payload_kg, opt_large.range_m, opt_large.n_c, design_vars=tuple(x_opt))
        per_results.append(r if isinstance(r, dict) else None)

    # filter valid results
    valid = [p for p in per_results if isinstance(p, dict)]
    if len(valid) == 0:
        print('No valid MC samples to summarise.')
    else:
        # compute statistics for MTOM
        Wtot = np.array([p['W_total'] for p in valid])
        meanW = float(np.mean(Wtot))
        stdW = float(np.std(Wtot, ddof=0))
        p_lo, p_hi = np.percentile(Wtot/G, [2.5, 97.5])
        print('\n--- Robust Design Summary (at optimized design) ---')
        mean_res_dict = {}
        # average other fields
        keys = ['W_battery','W_empty','P_hover','P_cruise','V_inf','r','J','S_w','E_req','disk_loading','blade_loading','cruise_CL','weight_residual']
        for k in keys:
            vals = np.array([p[k] for p in valid])
            mean_res_dict[k] = float(np.mean(vals))
        b = float(np.sqrt(AR_FIXED * mean_res_dict['S_w']))
        chord = mean_res_dict['S_w'] / b
        mean_result = SizingResult(
            W_total=meanW, W_battery=mean_res_dict['W_battery'], W_empty=mean_res_dict['W_empty'],
            P_hover=mean_res_dict['P_hover'], P_cruise=mean_res_dict['P_cruise'],
            V_inf=mean_res_dict['V_inf'], r=mean_res_dict['r'], J=mean_res_dict['J'], S_w=mean_res_dict['S_w'],
            b=b, chord=chord, E_req=mean_res_dict['E_req'], converged=True,
            disk_loading=mean_res_dict['disk_loading'], blade_loading=mean_res_dict['blade_loading'], cruise_CL=mean_res_dict['cruise_CL'], weight_residual=mean_res_dict['weight_residual'],
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )
        print(mean_result.summary())
        print('\nMTOM (kg) — mean: {0:.3f}, std: {1:.3f}, 95% PI: [{2:.3f}, {3:.3f}]'.format(meanW/G, stdW/G, p_lo, p_hi))

        # plot histogram
        try:
            mtom_kg = Wtot / G
            plt.figure(figsize=(7,4))
            plt.hist(mtom_kg, bins=30, alpha=0.8, color='C0', density=True)
            plt.axvline(np.mean(mtom_kg), color='k', ls='--', label=f'Mean {np.mean(mtom_kg):.3f} kg')
            plt.axvline(p_lo, color='r', ls=':', label=f'2.5% {p_lo:.3f} kg')
            plt.axvline(p_hi, color='r', ls=':', label=f'97.5% {p_hi:.3f} kg')
            plt.xlabel('MTOM (kg)')
            plt.ylabel('Density')
            plt.legend()
            outpath = 'sizing_openmdao/robust_mtom_hist.png'
            plt.tight_layout()
            plt.savefig(outpath, dpi=200)
            print(f'Histogram saved to {outpath}')
        except Exception as e:
            print('Plotting failed:', e)
