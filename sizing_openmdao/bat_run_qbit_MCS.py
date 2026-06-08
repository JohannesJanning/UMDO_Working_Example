"""
run_qbit_robust.py - Robust sizing optimization for QBiT with uncertain battery specific energy.

Minimize U = 0.5 * mean(W_total) + 0.5 * std(W_total) where the uncertainty in
`BATTERY_DENSITY` is propagated via Monte Carlo sampling. For each Monte Carlo sample
an inner OpenMDAO solve finds the required `W_total` (weight closure) for the
given design variables.

Notes:
- This implements a nested (outer) optimizer over design variables
  [V_inf, r, J, S_w] and (inner) solves that compute `W_total` for sampled
  `BATTERY_DENSITY` values. The implementation is intentionally simple and
  sequential; Monte Carlo and inner solves are expensive.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import lognorm
from scipy.stats import qmc
from scipy.optimize import minimize, brentq
import seaborn as sns

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
import seaborn as sns


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


def get_lognormal_dist(median: float = 235.0, sigma_ln: float = 0.28):
    """
    Create a lognormal distribution for battery specific energy.
    
    Parameters:
    - median: median value in Wh/kg (default: 235 Wh/kg)
    - sigma_ln: logarithmic standard deviation (default: 0.28)
    
    Returns:
    - lognorm distribution object with scale = median
    """
    # For lognormal: scale = median
    return lognorm(s=sigma_ln, scale=median)


def sample_battery_density(n_samples: int, median: float = 235.0, sigma_ln: float = 0.28,
                           seed: int | None = None, method: str = "lhs") -> np.ndarray:
    """
    Generate samples for BATTERY_DENSITY using the chosen sampling method.
    
    Battery specific energy follows a lognormal distribution with:
    - median = 235 Wh/kg
    - 5th percentile ≈ 150 Wh/kg
    - 95th percentile ≈ 370 Wh/kg
    
    method: 'lhs' for Latin Hypercube Sampling (recommended) or 'mcs'
            for plain Monte Carlo sampling.
    Returns an array of shape (n_samples,) in Wh/kg.
    """
    dist = get_lognormal_dist(median, sigma_ln)
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


def inner_solve_for_Wtotal(rho_bat_sample: float, payload_kg: float, range_m: float, n_c: int,
                           design_vars: Sequence[float], w_initial: float = 6.0 * G) -> float:
    """For a single sampled `rho_bat` (battery specific energy in Wh/kg), 
    build the OpenMDAO problem with that battery density and the given fixed 
    design variables and solve only for `W_total` (the inner weight-closure solve). 
    Returns found W_total (N) or np.nan on failure.
    
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
        # set battery density and W_total, run model (not driver)
        orig_rho = getattr(sc, 'BATTERY_DENSITY', None)
        sc.BATTERY_DENSITY = float(rho_bat_sample)
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
            if orig_rho is None:
                delattr(sc, 'BATTERY_DENSITY')
            else:
                sc.BATTERY_DENSITY = orig_rho

    
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
        orig_rho = getattr(sc, 'BATTERY_DENSITY', None)
        sc.BATTERY_DENSITY = float(rho_bat_sample)
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
            if orig_rho is None:
                delattr(sc, 'BATTERY_DENSITY')
            else:
                sc.BATTERY_DENSITY = orig_rho
    except Exception:
        # fallback: try running the original inner optimizer as a last resort
        try:
            orig_rho = getattr(sc, 'BATTERY_DENSITY', None)
            sc.BATTERY_DENSITY = float(rho_bat_sample)
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
    median_rho: float = 235.0      # median battery specific energy (Wh/kg)
    sigma_ln: float = 0.28         # logarithmic standard deviation
    seed: int | None = 123
    mc_samples: np.ndarray | None = None
    sampling_method: str = "lhs"
    cl_margin: float = 0.0000  # small margin to ensure we don't optimize right up to the CL_MAX constraint
    _obj_calls: int = 0
    eval_times: list | None = None
    inner_calls: int = 0
    inner_time_total: float = 0.0
    n_jobs: int = 2
    maxiter: int = 200

    def objective(self, x: Sequence[float]) -> float:
        """Outer objective to minimize: U = mean(W) + 1.96 * std(W)
        This corresponds to minimizing the 97.5th percentile (upper bound of 95% CI).
        """
        V_inf, r, J, S_w = x
        t0 = time.time()
        stats = self._mc_stats(x)
        t1 = time.time()
        
        # record timing
        if self.eval_times is None:
            self.eval_times = []
        self.eval_times.append(t1 - t0)
        self._obj_calls += 1
        
        # progress bar logic
        bar_len = 20
        avg = float(np.mean(self.eval_times)) if self.eval_times else float('nan')
        remaining = max(0, int(self.maxiter - self._obj_calls))
        eta_s = avg * remaining
        
        try:
            hrs, mins = int(eta_s // 3600), int((eta_s % 3600) // 60)
            secs = int(eta_s % 60)
            eta_str = f"{hrs}h{mins}m{secs}s"
        except:
            eta_str = "--"

        frac = min(1.0, float(self._obj_calls) / float(self.maxiter)) if self.maxiter > 0 else 0.0
        bar = "[" + "#" * int(frac * bar_len) + " " * (bar_len - int(frac * bar_len)) + "]"

        if stats is None:
            print(f"\r{bar} [{self._obj_calls}/{self.maxiter}] ETA {eta_str} → FAILED", end="")
            return 1e6

        meanW = stats['meanW']
        stdW = stats['stdW']

        # Use 1.96 for 97.5th percentile (upper bound of 95% CI)
        # This matches the 97.5th percentile used in post-processing
        p97_5_W = meanW + 2 * stdW

        # Objective: minimize the estimated 97.5th-percentile W (in N)
        U = float(p97_5_W)

        line = (f"{bar} [{self._obj_calls}/{self.maxiter}] ETA {eta_str} "
            f"→ V={V_inf:.2f}, r={r:.3f}, S_w={S_w:.3f} "
            f"mean={meanW/G:.2f}kg, p97.5={p97_5_W/G:.2f}kg")
        print('\r' + line, end="", flush=True)
        
        self._last_results = stats['results']
        return U

    def _mc_stats(self, x: Sequence[float]):
        """Evaluate MC samples for design x and return aggregated stats.
        Returns dict with keys: meanW, stdW, mean_metrics_dict, results (list), or
        None on failure.
        """
        V_inf, r, J, S_w = x
        if self.mc_samples is None:
            samples = sample_battery_density(self.n_mc, self.median_rho, self.sigma_ln,
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
                def _call_inner(rho_val):
                    return inner_solve_for_Wtotal(rho_val, self.payload_kg, self.range_m, self.n_c,
                                                  design_vars=(V_inf, r, J, S_w))

                res_list = Parallel(n_jobs=workers, prefer='processes')(delayed(_call_inner)(rho) for rho in samples)
            except Exception:
                # fallback to sequential if parallelization fails
                res_list = [inner_solve_for_Wtotal(rho, self.payload_kg, self.range_m, self.n_c,
                                                   design_vars=(V_inf, r, J, S_w)) for rho in samples]
        else:
            res_list = [inner_solve_for_Wtotal(rho, self.payload_kg, self.range_m, self.n_c,
                                               design_vars=(V_inf, r, J, S_w)) for rho in samples]

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

        # 1. Standard Statistics
        meanW = float(np.mean(W_arr))
        stdW = float(np.std(W_arr, ddof=0))
        
        # 2. 97.5th Percentile (matches objective)
        p97_5_W = meanW + 1.96 * stdW

        # 3. Also compute sample-based 97.5th percentile for validation
        p97_5_sample = float(np.percentile(W_arr, 97.5))

        # 4. Aggregate other fields
        keys = ['W_battery','W_empty','P_hover','P_cruise','V_inf','r','J','S_w','E_req','disk_loading','blade_loading','cruise_CL','weight_residual']
        mean_res = {}
        for k in keys:
            vals = np.array([p[k] for p in results if isinstance(p, dict)])
            if vals.size > 0:
                mean_res[k] = float(np.mean(vals))
            else:
                mean_res[k] = float('nan')

        return {
            'meanW': meanW, 
            'stdW': stdW, 
            'p97_5W': p97_5_W,     
            'p97_5_sample': p97_5_sample,
            'mean_res': mean_res, 
            'results': results,
            'W_samples': W_arr  
        }

    def run(self, x0: Sequence[float] | None = None, method: str = 'SLSQP'):
        # bounds
        bounds = [V_INF_BOUNDS, R_BOUNDS, J_BOUNDS, S_W_BOUNDS]
        if x0 is None:
            x0 = [np.mean(b) for b in bounds]
        # Pre-generate common MC samples (CRN) for all objective evaluations
        self.mc_samples = sample_battery_density(self.n_mc, self.median_rho, self.sigma_ln,
                                                 seed=self.seed, method=self.sampling_method)
        print("Starting robust optimization (using common random numbers).")
        print(f"Battery specific energy uncertainty: Lognormal(median={self.median_rho} Wh/kg, σ_ln={self.sigma_ln})")
        
        def constr_cruise_CL(x):
            stats = self._mc_stats(x)
            if stats is None:
                return -1e6
            return float(CL_MAX - stats['mean_res']['cruise_CL'])
        
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
    # Battery specific energy uncertainty parameters
    # Lognormal with median 235 Wh/kg, σ_ln = 0.28
    # Corresponds to 5th percentile ~150 Wh/kg, 95th ~370 Wh/kg
    MEDIAN_RHO = 235.0
    SIGMA_LN = 0.28
    
    opt = RobustOptimizer(payload_kg=3.0, range_m=15_000.0, n_c=2, n_mc=100, 
                         median_rho=MEDIAN_RHO, sigma_ln=SIGMA_LN, seed=123)
    x0 = [28.70, 0.2678, 1.3, 0.2296]
    res = opt.run(x0=x0, method='SLSQP')
    print('\nOptimization finished:')
    print(res)

    # Post-process: evaluate optimized design with larger MC and print styled summary
    x_opt = res.x
    opt_large = RobustOptimizer(payload_kg=opt.payload_kg, range_m=opt.range_m, n_c=opt.n_c,
                                n_mc=2000, median_rho=opt.median_rho, sigma_ln=opt.sigma_ln, 
                                seed=opt.seed, sampling_method=opt.sampling_method)
    samples = opt_large.mc_samples = sample_battery_density(opt_large.n_mc, opt_large.median_rho, 
                                                            opt_large.sigma_ln, seed=opt_large.seed, 
                                                            method=opt_large.sampling_method)
    per_results = []
    for rho in samples:
        r = inner_solve_for_Wtotal(rho, opt_large.payload_kg, opt_large.range_m, opt_large.n_c, 
                                   design_vars=tuple(x_opt))
        per_results.append(r if isinstance(r, dict) else None)

    # Print battery density statistics for context
    print(f"\n--- Battery Specific Energy Distribution ---")
    print(f"Lognormal(median={MEDIAN_RHO} Wh/kg, σ_ln={SIGMA_LN})")
    print(f"  5th percentile: {np.percentile(samples, 5):.1f} Wh/kg")
    print(f"  95th percentile: {np.percentile(samples, 95):.1f} Wh/kg")
    print(f"  Mean: {np.mean(samples):.1f} Wh/kg")
    print(f"  Std: {np.std(samples):.1f} Wh/kg")

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
            disk_loading=mean_res_dict['disk_loading'], blade_loading=mean_res_dict['blade_loading'], 
            cruise_CL=mean_res_dict['cruise_CL'], weight_residual=mean_res_dict['weight_residual'],
            DL_MAX=DL_MAX, BL_MAX=BL_MAX, CL_MAX=CL_MAX
        )
        print(mean_result.summary())
        print('\nMTOM (kg) — mean: {0:.3f}, std: {1:.3f}, 95% PI: [{2:.3f}, {3:.3f}]'.format(meanW/G, stdW/G, p_lo, p_hi))

        # plot histogram (high-fidelity, improved aesthetics)
        try:
            mtom_kg = Wtot / G
            plt.figure(figsize=(10,5), dpi=300)
            sns.histplot(mtom_kg, bins=100, stat='density', color='C0', kde=True)
            mean_val = np.mean(mtom_kg)
            p_lo, p_hi = np.percentile(mtom_kg, [2.5,97.5])
            plt.axvline(mean_val, color='k', ls='--', label=f'Mean {mean_val:.3f} kg')
            plt.axvline(p_lo, color='r', ls=':', label=f'2.5% {p_lo:.3f} kg')
            plt.axvline(p_hi, color='r', ls=':', label=f'97.5% {p_hi:.3f} kg')
            plt.xlabel('MTOM (kg)')
            plt.ylabel('Density')
            plt.title(f'Robust Design MTOM Distribution\nBattery Density Uncertainty: Lognormal(median={MEDIAN_RHO} Wh/kg, σ={SIGMA_LN})')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            outpath_png = 'sizing_openmdao/robust_mtom_hist.png'
            outpath_svg = 'sizing_openmdao/robust_mtom_hist.svg'
            plt.savefig(outpath_png, dpi=300)
            plt.savefig(outpath_svg)
            print(f'Histogram saved to {outpath_png} and {outpath_svg}')
        except Exception as e:
            print('Plotting failed:', e)