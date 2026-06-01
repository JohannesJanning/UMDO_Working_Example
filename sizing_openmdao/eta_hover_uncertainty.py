import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def get_truncnorm_params(mean, std, low, high):
    """
    Calculates parameters for truncated normal distribution.
    Returns (a, b, loc, scale) for scipy.stats.truncnorm
    """
    # Standard normal bounds
    a = (low - mean) / std
    b = (high - mean) / std
    return a, b, mean, std

# --- CONFIGURATION (Epistemic Uncertainty for ETA_HOVER) ---
ETA_MEAN = 0.65      # Nominal value from literature
ETA_STD  = 0.05     # Epistemic uncertainty magnitude
ETA_LO   = 0.55      # Physical lower bound (realistic minimum)
ETA_HI   = 0.75      # Physical upper bound (theoretical maximum)

# Get truncated normal parameters
a, b, loc, scale = get_truncnorm_params(ETA_MEAN, ETA_STD, ETA_LO, ETA_HI)
dist = truncnorm(a, b, loc=loc, scale=scale)

# --- EXACT PERCENTILE BOUNDS (alpha=0.05) ---
p_lower = 0.025
p_upper = 0.975
bound_lower = dist.ppf(p_lower)
bound_upper = dist.ppf(p_upper)

# --- PLOTTING ---
x = np.linspace(0.50, 0.80, 1000)
pdf = dist.pdf(x)
cdf = dist.cdf(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. PDF PLOT: 2.5% to 97.5% Range
ax1.plot(x, pdf, color='steelblue', lw=2.5, label='Hover FOM PDF')
ax1.fill_between(x, pdf, where=(x >= bound_lower) & (x <= bound_upper), 
                 color='steelblue', alpha=0.15, label='95% Credible Interval (2.5% - 97.5%)')

# Vertical lines for bounds
ax1.axvline(bound_lower, color='navy', ls=':', lw=1.5, label=f'2.5%: {bound_lower:.3f}')
ax1.axvline(bound_upper, color='navy', ls=':', lw=1.5, label=f'97.5%: {bound_upper:.3f}')
ax1.axvline(ETA_MEAN, color='darkorange', ls='-', alpha=0.6, label=f'Nominal: {ETA_MEAN:.3f}')

# Add physical bounds as vertical lines
ax1.axvline(ETA_LO, color='gray', ls='--', alpha=0.4, label=f'Physical bound: {ETA_LO:.2f}')
ax1.axvline(ETA_HI, color='gray', ls='--', alpha=0.4, label=f'Physical bound: {ETA_HI:.2f}')

ax1.set_title('Probability Density Function (PDF) - Epistemic Uncertainty', fontweight='bold')
ax1.set_xlabel('Rotor Figure of Merit $η_{hover}$ [-]')
ax1.set_ylabel('Density')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# 2. CDF PLOT: Visualizing the "Tails"
ax2.plot(x, cdf, color='seagreen', lw=2.5, label='Hover FOM CDF')

# Shade the 95% probability mass on the CDF
ax2.fill_betweenx([p_lower, p_upper], bound_lower, bound_upper, color='seagreen', alpha=0.1)

# Highlight specific probability thresholds
ax2.axhline(p_lower, color='grey', ls='--', lw=1)
ax2.axhline(p_upper, color='grey', ls='--', lw=1)
ax2.axvline(ETA_MEAN, color='darkorange', ls='-', lw=1.5, label=f'Nominal: {ETA_MEAN:.3f}')

# Annotate the interval on the CDF
ax2.annotate('', xy=(bound_lower, 0.5), xytext=(bound_upper, 0.5),
             arrowprops=dict(arrowstyle='<->', color='darkgreen'))
ax2.text((bound_lower+bound_upper)/2, 0.52, '95% Credible Interval', 
         ha='center', color='darkgreen', fontweight='bold')

ax2.set_title('Cumulative Distribution Function (CDF) - Epistemic Uncertainty', fontweight='bold')
ax2.set_xlabel('Rotor Figure of Merit $η_{hover}$ [-]')
ax2.set_ylabel('Cumulative Probability $P(η \\leq value)$')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- ADDITIONAL CLEAN PDF PLOT ---
fig_clean, ax_clean = plt.subplots(figsize=(8, 5))

# PDF curve
ax_clean.plot(x, pdf, color='steelblue', lw=2.5)

# 95% interval shading
ax_clean.fill_between(
    x,
    pdf,
    where=(x >= bound_lower) & (x <= bound_upper),
    color='steelblue',
    alpha=0.15
)

# Percentile bounds
ax_clean.axvline(bound_lower, color='navy', ls=':', lw=1.5)
ax_clean.axvline(bound_upper, color='navy', ls=':', lw=1.5)

# Nominal value
ax_clean.axvline(ETA_MEAN, color='darkorange', ls='-', lw=1.8, alpha=0.7)

# Physical bounds (light gray)
ax_clean.axvline(ETA_LO, color='gray', ls='--', lw=1, alpha=0.5)
ax_clean.axvline(ETA_HI, color='gray', ls='--', lw=1, alpha=0.5)

# Labels and styling
ax_clean.set_title('Hover Figure of Merit PDF (Epistemic Uncertainty)', fontweight='bold', fontsize=16)
ax_clean.set_xlabel('Rotor Figure of Merit $η_{hover}$ [-]', fontsize=14)
ax_clean.set_ylabel('Density', fontsize=14)
ax_clean.tick_params(axis='both', labelsize=12)

# Add text annotation for bounds
ax_clean.text(bound_lower - 0.008, ax_clean.get_ylim()[1]*0.9, 
              f'2.5%\n{bound_lower:.3f}', ha='center', fontsize=9, color='navy')
ax_clean.text(bound_upper + 0.008, ax_clean.get_ylim()[1]*0.9, 
              f'97.5%\n{bound_upper:.3f}', ha='center', fontsize=9, color='navy')

ax_clean.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- PRINT STATEMENTS FOR DOCUMENTATION ---
print("=" * 65)
print("Epistemic Uncertainty Characterization for Rotor FOM")
print("=" * 65)
print(f"Distribution: Truncated Normal")
print(f"Nominal (mean): {ETA_MEAN:.3f}")
print(f"Epistemic std:  {ETA_STD:.3f}")
print(f"Physical bounds: [{ETA_LO:.2f}, {ETA_HI:.2f}]")
print()
print(f"--- 95% Credible Interval ---")
print(f"Lower bound (2.5th percentile): {bound_lower:.4f}")
print(f"Upper bound (97.5th percentile): {bound_upper:.4f}")
print(f"Interval width: {bound_upper - bound_lower:.4f}")
print()
print(f"--- Probability of Interest ---")
print(f"P(η < 0.60): {(dist.cdf(0.60))*100:.2f}%")
print(f"P(η > 0.70): {(1 - dist.cdf(0.70))*100:.2f}%")
print(f"P(η < 0.55): {(dist.cdf(0.55))*100:.2f}%  (truncated)")
print(f"P(η > 0.75): {(1 - dist.cdf(0.75))*100:.2f}%  (truncated)")
print("=" * 65)