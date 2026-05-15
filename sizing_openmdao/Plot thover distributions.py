"""
Comparison of t_hover uncertainty distributions:
  1. Shifted lognormal (used in MC robust optimizer)
  2. Beta[25,130] percentile-matched approximation (used in UQPCE PCE surrogate)

Produces a 2x2 figure:
  Top-left:  PDF overlay
  Top-right: CDF overlay
  Bottom-left:  Difference in PDF (Beta - Lognormal)
  Bottom-right: Quantile-quantile comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import lognorm, beta
from scipy.integrate import quad

# ── Distribution parameters ──────────────────────────────────────────────────
T_MEAN = 55.0
T_STD  = 18.0
T_MIN  = 25.0

# Shifted lognormal
mu_prime = T_MEAN - T_MIN
v_prime  = T_STD**2
s_sq     = np.log(v_prime / mu_prime**2 + 1)
s_ln     = np.sqrt(s_sq)
scale_ln = mu_prime * np.exp(-s_sq / 2)
dist_ln  = lognorm(s=s_ln, loc=T_MIN, scale=scale_ln * np.exp(s_sq / 2))

# Truncation bounds
A, B = 25.0, 130.0
C = dist_ln.cdf(B) - dist_ln.cdf(A)          # normalisation constant

def trunc_ln_pdf(x):
    return np.where((x >= A) & (x <= B), dist_ln.pdf(x) / C, 0.0)

def trunc_ln_cdf(x):
    return np.clip((dist_ln.cdf(np.maximum(x, A)) - dist_ln.cdf(A)) / C, 0, 1)

# Beta[25,130] — percentile-matched fit
ALPHA_B, BETA_B = 1.5774, 4.0257
dist_b = beta(ALPHA_B, BETA_B, loc=A, scale=B - A)

# ── Derived statistics ────────────────────────────────────────────────────────
x_plot = np.linspace(20, 145, 2000)

# Lognormal (full, untruncated) percentiles for annotation
p_lo_full = dist_ln.ppf(0.025)
p_hi_full = dist_ln.ppf(0.975)
p_mean_ln = dist_ln.mean()

# Truncated lognormal percentiles
def trunc_ppf(p):
    return dist_ln.ppf(dist_ln.cdf(A) + p * C)

p25_ln  = trunc_ppf(0.25);   p75_ln  = trunc_ppf(0.75)
p95_ln  = trunc_ppf(0.95);   p975_ln = trunc_ppf(0.975)
mean_trunc = quad(lambda x: x * trunc_ln_pdf(x), A, B)[0]

# Beta percentiles
p25_b   = dist_b.ppf(0.25);  p75_b   = dist_b.ppf(0.75)
p95_b   = dist_b.ppf(0.95);  p975_b  = dist_b.ppf(0.975)

# ── Colours ───────────────────────────────────────────────────────────────────
C_LN  = '#C0392B'      # firebrick — lognormal
C_LNT = '#E74C3C'      # lighter   — truncated region
C_B   = '#2471A3'      # steel blue — Beta
C_BF  = '#5DADE2'      # lighter   — Beta fill
GREY  = '#7F8C8D'

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    r'$t_{\mathrm{hover}}$ Uncertainty Distribution: Lognormal vs Beta Approximation',
    fontsize=14, fontweight='bold', y=0.98
)

# ─── (0,0) PDF overlay ───────────────────────────────────────────────────────
ax = axes[0, 0]

# Full lognormal (faint)
ax.plot(x_plot, dist_ln.pdf(x_plot), color=C_LN, lw=1.2, ls='--', alpha=0.4,
        label='Lognormal (untruncated)')

# Truncated lognormal
ax.plot(x_plot, trunc_ln_pdf(x_plot), color=C_LN, lw=2.5,
        label=f'Lognormal truncated [{A:.0f},{B:.0f}]s')
ax.fill_between(x_plot, trunc_ln_pdf(x_plot),
                where=(x_plot >= p25_ln) & (x_plot <= p75_ln),
                color=C_LN, alpha=0.12, label='IQR lognormal')

# Beta
ax.plot(x_plot, dist_b.pdf(x_plot), color=C_B, lw=2.5,
        label=f'Beta({ALPHA_B},{BETA_B}) on [{A},{B}]s')
ax.fill_between(x_plot, dist_b.pdf(x_plot),
                where=(x_plot >= p25_b) & (x_plot <= p75_b),
                color=C_B, alpha=0.12, label='IQR Beta')

# Annotations
for xv, col, txt in [
    (mean_trunc,  C_LN, f'μ={mean_trunc:.1f}s'),
    (dist_b.mean(), C_B, f'μ={dist_b.mean():.1f}s'),
]:
    ax.axvline(xv, color=col, ls=':', lw=1.4, alpha=0.8)
    ax.text(xv + 0.8, ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else 0.001,
            txt, color=col, fontsize=8, va='bottom')

ax.axvline(B, color=GREY, ls='-.', lw=1.2, alpha=0.7,
           label=f'Truncation bound {B:.0f}s')
ax.set_xlim(18, 145)
ax.set_xlabel(r'Hover time $t_{\mathrm{hover}}$ [s]', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('PDF comparison', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, ls='--', alpha=0.4)

# ─── (0,1) CDF overlay ───────────────────────────────────────────────────────
ax = axes[0, 1]

cdf_ln_vals = trunc_ln_cdf(x_plot)
cdf_b_vals  = dist_b.cdf(x_plot)

ax.plot(x_plot, cdf_ln_vals, color=C_LN, lw=2.5,
        label='Truncated lognormal CDF')
ax.plot(x_plot, cdf_b_vals,  color=C_B,  lw=2.5, ls='--',
        label=f'Beta CDF')

# 97.5th percentile markers
ax.axhline(0.975, color=GREY, ls=':', lw=1.2)
ax.axhline(0.025, color=GREY, ls=':', lw=1.2)

for xv, col, lbl in [
    (p975_ln, C_LN, f'p97.5={p975_ln:.1f}s'),
    (p975_b,  C_B,  f'p97.5={p975_b:.1f}s'),
]:
    ax.axvline(xv, color=col, ls=':', lw=1.4, alpha=0.8)
    ax.annotate(lbl, xy=(xv, 0.975), xytext=(xv - 18, 0.88),
                color=col, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

# UQPCE CI target band
ax.axhspan(0.025, 0.975, color='gold', alpha=0.06,
           label='UQPCE 95% CI target band')

ax.set_xlim(18, 145)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r'Hover time $t_{\mathrm{hover}}$ [s]', fontsize=11)
ax.set_ylabel(r'Cumulative probability $P(T \leq t)$', fontsize=11)
ax.set_title('CDF comparison', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, ls='--', alpha=0.4)

# ─── (1,0) PDF difference ────────────────────────────────────────────────────
ax = axes[1, 0]

x_inner = np.linspace(A + 0.1, B - 0.1, 1000)
diff = dist_b.pdf(x_inner) - trunc_ln_pdf(x_inner)

ax.axhline(0, color='k', lw=0.8)
ax.fill_between(x_inner, diff, where=(diff > 0), color=C_B,  alpha=0.35,
                label='Beta > Lognormal')
ax.fill_between(x_inner, diff, where=(diff < 0), color=C_LN, alpha=0.35,
                label='Lognormal > Beta')
ax.plot(x_inner, diff, color='#2C3E50', lw=1.5)

ax.axvline(B, color=GREY, ls='-.', lw=1.0, alpha=0.7)
ax.set_xlim(18, 145)
ax.set_xlabel(r'Hover time $t_{\mathrm{hover}}$ [s]', fontsize=11)
ax.set_ylabel(r'$f_{\mathrm{Beta}}(t) - f_{\mathrm{LN,trunc}}(t)$', fontsize=11)
ax.set_title('PDF difference (Beta − truncated lognormal)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, ls='--', alpha=0.4)

# Annotation: area of over/underestimation
pos_area = np.trapezoid(np.maximum(diff, 0), x_inner)
neg_area = np.trapezoid(np.minimum(diff, 0), x_inner)
ax.text(0.03, 0.92, f'Beta overestimates: {pos_area:.3f}\nBeta underestimates: {abs(neg_area):.3f}',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GREY, alpha=0.8))

# ─── (1,1) Quantile–quantile + key statistics table ──────────────────────────
ax = axes[1, 1]

probs = np.linspace(0.01, 0.99, 200)
q_ln = np.array([trunc_ppf(p) for p in probs])
q_b  = dist_b.ppf(probs)

ax.plot([A, B], [A, B], color=GREY, lw=1.2, ls='--', label='Perfect agreement')
ax.plot(q_ln, q_b, color='#8E44AD', lw=2.2, label='Beta vs Lognormal quantiles')

# Highlight 97.5th percentile
ax.scatter([p975_ln], [p975_b], color='gold', s=80, zorder=5,
           edgecolors='k', lw=0.8, label=f'p97.5: LN={p975_ln:.1f}s, B={p975_b:.1f}s')
ax.scatter([p975_ln], [p975_ln], color=C_LN, s=60, zorder=5,
           edgecolors='k', lw=0.8, marker='D')

ax.set_xlabel('Truncated lognormal quantile [s]', fontsize=11)
ax.set_ylabel('Beta quantile [s]', fontsize=11)
ax.set_title('Q–Q plot: Beta vs truncated lognormal', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, ls='--', alpha=0.4)

# Statistics text box
stats = (
    f"{'':─<38}\n"
    f"  {'':20s} {'LN trunc':>8}  {'Beta':>8}\n"
    f"  {'Mean [s]':20s} {mean_trunc:>8.2f}  {dist_b.mean():>8.2f}\n"
    f"  {'Std [s]':20s} {quad(lambda x:(x-mean_trunc)**2*trunc_ln_pdf(x),A,B)[0]**0.5:>8.2f}  {dist_b.std():>8.2f}\n"
    f"  {'p25 [s]':20s} {p25_ln:>8.2f}  {p25_b:>8.2f}\n"
    f"  {'p75 [s]':20s} {p75_ln:>8.2f}  {p75_b:>8.2f}\n"
    f"  {'p95 [s]':20s} {p95_ln:>8.2f}  {p95_b:>8.2f}\n"
    f"  {'p97.5 [s]':20s} {p975_ln:>8.2f}  {p975_b:>8.2f}\n"
    f"  {'Prob mass [25,130]':20s} {C*100:>7.2f}%  {'100.00%':>8}\n"
    f"{'':─<38}"
)
ax.text(1.04, 0.5, stats, transform=ax.transAxes, fontsize=7.5,
        va='center', ha='left', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#F8F9FA', ec=GREY, alpha=0.95))

plt.tight_layout(rect=[0, 0, 0.87, 0.97])

out_path = '/Users/johannesjanning/Library/CloudStorage/Dropbox/Mac (3)/Documents/Research Coding/MDO-FSMVRP UMDO/sizing_openmdao/thover_distribution_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved to {out_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  t_hover distribution summary")
print(f"{'─'*55}")
print(f"  {'':22s} {'LN (full)':>10}  {'LN trunc':>10}  {'Beta':>10}")
print(f"  {'Mean [s]':22s} {dist_ln.mean():>10.2f}  {mean_trunc:>10.2f}  {dist_b.mean():>10.2f}")
print(f"  {'Std [s]':22s} {dist_ln.std():>10.2f}  {quad(lambda x:(x-mean_trunc)**2*trunc_ln_pdf(x),A,B)[0]**0.5:>10.2f}  {dist_b.std():>10.2f}")
print(f"  {'p2.5 [s]':22s} {p_lo_full:>10.2f}  {'—':>10}  {dist_b.ppf(0.025):>10.2f}")
print(f"  {'p97.5 [s]':22s} {p_hi_full:>10.2f}  {p975_ln:>10.2f}  {p975_b:>10.2f}")
print(f"  {'Support':22s} {'[25, ∞)':>10}  {'[25,130]':>10}  {'[25,130]':>10}")
print(f"  {'Mass in [25,130]':22s} {C*100:>9.2f}%  {'100.00%':>10}  {'100.00%':>10}")
print(f"  {'p97.5 error vs LN trunc':22s} {'':>10}  {'—':>10}  {p975_b-p975_ln:>+10.2f}s")
print(f"{'─'*55}")
print(f"\n  UQPCE config: Beta(alpha={ALPHA_B}, beta={BETA_B}),")
print(f"  interval_low=25, interval_high=130")
print(f"  Fitted by percentile matching (p50,p75,p90,p95,p975)")