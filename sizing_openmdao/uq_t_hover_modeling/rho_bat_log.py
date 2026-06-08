import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def get_lognormal_params_from_percentiles(median, p5, p95):
    """
    Calculates lognormal parameters (s, scale) from median and 5th/95th percentiles.
    For lognormal: median = scale, and s = ln(p95/median)/norm.ppf(0.95)
    """
    from scipy.stats import norm
    s = np.log(p95 / median) / norm.ppf(0.95)
    scale = median  # for lognormal, scale = median
    return s, scale

# --- CONFIGURATION (Battery Specific Energy Uncertainty) ---
# Based on: median=235 Wh/kg, 5th=150 Wh/kg, 95th=370 Wh/kg
MEDIAN = 235.0      # Wh/kg
P5 = 150.0          # 5th percentile (Wh/kg)
P95 = 370.0         # 95th percentile (Wh/kg)

# Alternative: direct lognormal parameters (μ, σ) from the paper
# ρ ~ Lognormal(μ=5.46, σ=0.28²) where μ and σ are parameters of underlying normal
MU_LOG = 5.46       # mean of log(ρ)
SIGMA_LOG = 0.28    # std dev of log(ρ)

# Two ways to define the distribution:
# Method 1: Using median and percentiles
s_percentile, scale_percentile = get_lognormal_params_from_percentiles(MEDIAN, P5, P95)
dist_percentile = lognorm(s_percentile, scale=scale_percentile)

# Method 2: Using log-space parameters (recommended, cleaner)
dist = lognorm(s=SIGMA_LOG, scale=np.exp(MU_LOG))  # scale = exp(μ)

# Verify the distribution matches the intended percentiles
computed_p5 = dist.ppf(0.05)
computed_p95 = dist.ppf(0.95)
computed_median = dist.median()

print(f"--- Distribution Verification ---")
print(f"Target median: {MEDIAN:.0f} Wh/kg → Computed: {computed_median:.1f} Wh/kg")
print(f"Target 5th: {P5:.0f} Wh/kg → Computed: {computed_p5:.1f} Wh/kg")
print(f"Target 95th: {P95:.0f} Wh/kg → Computed: {computed_p95:.1f} Wh/kg")

# --- EXACT PERCENTILE BOUNDS (alpha=0.05) ---
p_lower = 0.025
p_upper = 0.975
p_99 = 0.99
p_95_lower = 0.05
p_95_upper = 0.95

bound_99 = dist.ppf(p_99)
bound_975_lower = dist.ppf(p_lower)
bound_975_upper = dist.ppf(p_upper)
bound_95_lower = dist.ppf(p_95_lower)
bound_95_upper = dist.ppf(p_95_upper)

# --- PLOTTING ---
x = np.linspace(50, 550, 1000)
pdf = dist.pdf(x)
cdf = dist.cdf(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. PDF PLOT: 5% to 95% Range (primary interval)
ax1.plot(x, pdf, color='firebrick', lw=2.5, label='Battery Specific Energy PDF')
ax1.fill_between(x, pdf, where=(x >= bound_95_lower) & (x <= bound_95_upper), 
                 color='firebrick', alpha=0.15, label='90% Interval (5% - 95%)')

# Vertical lines for bounds
ax1.axvline(bound_95_lower, color='darkred', ls=':', lw=1.5, label=f'5%: {bound_95_lower:.0f} Wh/kg')
ax1.axvline(bound_95_upper, color='darkred', ls=':', lw=1.5, label=f'95%: {bound_95_upper:.0f} Wh/kg')
ax1.axvline(MEDIAN, color='blue', ls='-', alpha=0.6, label=f'Median: {MEDIAN:.0f} Wh/kg')

ax1.set_title('Probability Density Function (PDF)\nBattery Specific Energy', fontweight='bold')
ax1.set_xlabel('Specific Energy $\\rho_{bat}$ [Wh/kg]')
ax1.set_ylabel('Density')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# 2. CDF PLOT: Visualizing the "Tails"
ax2.plot(x, cdf, color='seagreen', lw=2.5, label='Battery Specific Energy CDF')

# Shade the 90% probability mass on the CDF
ax2.fill_betweenx([p_95_lower, p_95_upper], bound_95_lower, bound_95_upper, color='seagreen', alpha=0.1)

# Highlight specific probability thresholds
ax2.axhline(p_95_lower, color='grey', ls='--', lw=1)
ax2.axhline(p_95_upper, color='grey', ls='--', lw=1)
ax2.axvline(250, color='black', ls='-', lw=1.5, label='Nominal Design (250 Wh/kg)')

# Annotate the interval on the CDF
ax2.annotate('', xy=(bound_95_lower, 0.5), xytext=(bound_95_upper, 0.5),
             arrowprops=dict(arrowstyle='<->', color='darkgreen'))
ax2.text((bound_95_lower+bound_95_upper)/2, 0.52, '90% of Technology Outcomes', 
         ha='center', color='darkgreen', fontweight='bold')

ax2.set_title('Cumulative Distribution Function (CDF)', fontweight='bold')
ax2.set_xlabel('Specific Energy $\\rho_{bat}$ [Wh/kg]')
ax2.set_ylabel('Cumulative Probability $P(\\rho \\leq r)$')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- ADDITIONAL CLEAN PDF PLOT ---
fig_clean, ax_clean = plt.subplots(figsize=(8, 5))

# PDF curve
ax_clean.plot(x, pdf, color='firebrick', lw=2.5)

# 90% interval shading (5-95%)
ax_clean.fill_between(
    x,
    pdf,
    where=(x >= bound_95_lower) & (x <= bound_95_upper),
    color='firebrick',
    alpha=0.15
)

# Also show 95% interval (2.5-97.5%) as dashed lines for context
ax_clean.axvline(bound_975_lower, color='darkred', ls=':', lw=1, alpha=0.5)
ax_clean.axvline(bound_975_upper, color='darkred', ls=':', lw=1, alpha=0.5)

# Percentile bounds (primary 5-95%)
ax_clean.axvline(bound_95_lower, color='darkred', ls='--', lw=1.5, label=f'5th: {bound_95_lower:.0f} Wh/kg')
ax_clean.axvline(bound_95_upper, color='darkred', ls='--', lw=1.5, label=f'95th: {bound_95_upper:.0f} Wh/kg')
ax_clean.axvline(MEDIAN, color='blue', ls='-', lw=1.5, alpha=0.7, label=f'Median: {MEDIAN:.0f} Wh/kg')

# Labels and styling
ax_clean.set_title('Battery Specific Energy PDF\nTechnology Uncertainty at Conceptual Design Stage', 
                   fontweight='bold', fontsize=16)
ax_clean.set_xlabel('Specific Energy $\\rho_{bat}$ [Wh/kg]', fontsize=14)
ax_clean.set_ylabel('Density', fontsize=14)
ax_clean.tick_params(axis='both', labelsize=12)
ax_clean.legend(fontsize=10)
ax_clean.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- PRINT STATEMENTS FOR DOCUMENTATION ---
print(f"\n{'='*60}")
print(f"BATTERY SPECIFIC ENERGY UNCERTAINTY CHARACTERIZATION")
print(f"{'='*60}")
print(f"\nDistribution: Lognormal(μ={MU_LOG:.3f}, σ={SIGMA_LOG:.3f})")
print(f"  where μ, σ are parameters of ln(ρ_bat)")
print(f"\n--- Key Statistics ---")
print(f"Median: {dist.median():.1f} Wh/kg")
print(f"Mean: {dist.mean():.1f} Wh/kg")
print(f"Standard deviation: {dist.std():.1f} Wh/kg")
print(f"\n--- Percentiles ---")
print(f"5th percentile: {bound_95_lower:.1f} Wh/kg")
print(f"95th percentile: {bound_95_upper:.1f} Wh/kg")
print(f"90% interval width: {bound_95_upper - bound_95_lower:.1f} Wh/kg")
print(f"2.5th percentile: {bound_975_lower:.1f} Wh/kg")
print(f"97.5th percentile: {bound_975_upper:.1f} Wh/kg")
print(f"95% interval width: {bound_975_upper - bound_975_lower:.1f} Wh/kg")
print(f"\n--- 99th Percentile ---")
print(f"99th percentile: {bound_99:.1f} Wh/kg")
print(f"\n--- Design Implications ---")
print(f"Probability technology achieves <200 Wh/kg: {dist.cdf(200)*100:.1f}%")
print(f"Probability technology achieves <250 Wh/kg: {dist.cdf(250)*100:.1f}%")
print(f"Probability technology achieves >300 Wh/kg: {(1 - dist.cdf(300))*100:.1f}%")
print(f"Probability technology achieves >400 Wh/kg: {(1 - dist.cdf(400))*100:.1f}%")

print(f"\n{'='*60}")
print(f"BEST PAPER WORDING:")
print(f"{'='*60}")
print(f"Battery specific energy is modeled as a lognormal uncertainty with median "
      f"{MEDIAN:.0f} Wh/kg and logarithmic standard deviation {SIGMA_LOG:.2f}, "
      f"corresponding approximately to a 5–95% interval of {bound_95_lower:.0f}–{bound_95_upper:.0f} Wh/kg. "
      f"This representation reflects the positive, right-skewed, and multiplicative nature "
      f"of effective pack-level energy density at the conceptual design stage.")