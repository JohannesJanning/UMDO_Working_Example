import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# GENERIC GAUSSIAN DISTRIBUTION (Normalized)
# ============================================================================

# Parameters for a normalized Gaussian
mean = 0.5  # Shifted to center in [0,1] domain
std = 0.12  # Scaled to fit within [0,1]

# Generate x values and CDF
x = np.linspace(0, 1, 1000)
y_cdf = norm.cdf(x, loc=mean, scale=std)

# Key points for the generic example
p2_5 = norm.ppf(0.025, loc=mean, scale=std)   # 2.5th percentile
p97_5 = norm.ppf(0.975, loc=mean, scale=std)  # 97.5th percentile
p50 = norm.ppf(0.50, loc=mean, scale=std)     # median/mean

# CDF values at key points
cdf_at_mean = norm.cdf(mean, loc=mean, scale=std)     # 0.5
cdf_at_p2_5 = 0.025
cdf_at_p97_5 = 0.975

# Deterministic upper bound - shifted to the LEFT (larger failure region)
cert_value = 0.60  # Shifted left from 0.85 to 0.65

# Ensure bounds are within [0,1]
p2_5 = max(0, min(1, p2_5))
p97_5 = max(0, min(1, p97_5))
cert_value = max(0, min(1, cert_value))

# ============================================================================
# COMPUTE CR (now negative margin)
# ============================================================================
margin = cert_value - p97_5
uncertainty = p97_5 - mean
cr = margin / uncertainty if uncertainty > 0 else np.nan

print("\n" + "="*50)
print("Generic Gaussian Distribution Statistics:")
print("="*50)
print(f"  μ (mean)         : {mean:.3f} (CDF: {cdf_at_mean:.3f})")
print(f"  P₂.₅ (2.5%)      : {p2_5:.3f} (CDF: {cdf_at_p2_5:.3f})")
print(f"  P₅₀ (median)     : {p50:.3f}")
print(f"  P₉₇.₅ (97.5%)    : {p97_5:.3f} (CDF: {cdf_at_p97_5:.3f})")
print(f"  Deterministic bound: {cert_value:.3f}")
print(f"\n  Margin (M)       : {margin:.3f}")
print(f"  Uncertainty (U)  : {uncertainty:.3f}")
print(f"  CR = M/U         : {cr:.4f}")
print("="*50)

# ============================================================================
# MATPLOTLIB STYLE (EVEN LARGER text)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False
})

# ============================================================================
# FIGURE (square aspect ratio)
# ============================================================================
fig, ax = plt.subplots(figsize=(5.2, 5.2))  # Same width and height = square

# CDF curve (black)
ax.plot(x, y_cdf, color='black', lw=1.2)

# ============================================================================
# DOTS AT INTERSECTION POINTS (on the CDF curve)
# ============================================================================
# Removed the dot at p2_5 (2.5% point)
ax.plot(mean, cdf_at_mean, 'ko', markersize=6, markeredgewidth=0.5)
ax.plot(p97_5, cdf_at_p97_5, 'ko', markersize=6, markeredgewidth=0.5)

# ============================================================================
# PROBABILITY REQUIREMENT LINES
# ============================================================================
prob_req_high = 0.975
ax.axhline(y=prob_req_high, color='black', linestyle='--', lw=0.8)
ax.text(0.02, prob_req_high - 0.02, "Probability\nRequirement", 
        va='top', ha='left', fontsize=18)

prob_req_low = 0.025
ax.axhline(y=prob_req_low, color='black', linestyle='--', lw=0.8)

# ============================================================================
# VERTICAL LINES
# ============================================================================
ax.vlines(x=mean, ymin=0, ymax=cdf_at_mean, color='black', linestyle='--', lw=0.8)
# Removed vertical line at p2_5
ax.vlines(x=p97_5, ymin=0, ymax=prob_req_high, color='black', linestyle='--', lw=0.8)

# ============================================================================
# DETERMINISTIC BOUND LINE (red) - FULL HEIGHT from 0 to 1
# ============================================================================
ax.axvline(x=cert_value, ymin=0, ymax=1, color='red', lw=1.0)  # Full height from 0 to 1
# NO horizontal red line at the top (removed)

# Conditional shading for the margin area (now cert_value is left of p97_5)
if margin > 0:
    # Positive margin - light green shading
    ax.axvspan(p97_5, cert_value, ymin=0, ymax=prob_req_high, 
               alpha=0.15, color='green')
    margin_color = 'green'
elif margin < 0:
    # Negative margin - light red shading (cert_value < p97_5)
    ax.axvspan(cert_value, p97_5, ymin=0, ymax=prob_req_high, 
               alpha=0.15, color='red')
    margin_color = 'red'
else:
    margin_color = 'black'

# ============================================================================
# FAILURE REGION LABEL - Centered between p97_5 vertical line and right bound (1.0)
# ============================================================================
failure_center_x = (p97_5 + 1.0) / 2  # Centered between p97_5 and right edge
failure_center_y = prob_req_high / 2
ax.text(failure_center_x, failure_center_y, "Failure Region", color='red', 
        rotation=90, ha='center', va='center', fontsize=18)

# ============================================================================
# ARROWS (U and M)
# ============================================================================
# U arrow (uncertainty)
y_arrow_u = 0.25
ax.annotate('', xy=(mean, y_arrow_u), xytext=(p97_5, y_arrow_u),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
ax.text((mean + p97_5)/2, y_arrow_u + 0.02, 'U', 
        ha='center', va='bottom', fontsize=22, fontstyle='italic')

# M arrow (margin) - color-coded (now negative margin, so red)
y_arrow_m = 0.45
ax.annotate('', xy=(cert_value, y_arrow_m), xytext=(p97_5, y_arrow_m),
            arrowprops=dict(arrowstyle='<->', color=margin_color, lw=0.8))
ax.text((cert_value + p97_5)/2, y_arrow_m + 0.02, 'M', 
        ha='center', va='bottom', fontsize=22, fontstyle='italic', color=margin_color)

# ============================================================================
# X-AXIS (ticks only on bottom, pointing outward, no numbers, but frame visible)
# ============================================================================
ax.set_xlim(0, 1)

# Set ticks at 0, 0.1, 0.2, ..., 1.0 (clean, no 0.05 increments)
ax.set_xticks(np.arange(0, 1.05, 0.1))
ax.set_xticklabels(['', '', '', '', '', '', '', '', '', '', ''])  # No numbers

# Keep all spines (full frame/box)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Ticks pointing outward, only on bottom
ax.tick_params(axis='x', which='both', direction='out', 
               top=False, bottom=True, 
               length=6, width=0.8, labelsize=18)

ax.set_xlabel('Response value', fontsize=22)

# ============================================================================
# Y-AXIS (ticks only on left, pointing outward, numbers only at 0 and 1)
# ============================================================================
ax.set_ylim(0.0, 1.0)

# Set ticks at 0, 0.1, 0.2, ..., 1.0
ax.set_yticks(np.arange(0, 1.05, 0.1))

# Only label 0 and 1.0
y_labels = ['0'] + [''] * 9 + ['1.0']  # 0, then empty for 0.1-0.9, then 1.0 at the end
ax.set_yticklabels(y_labels)

# Ticks pointing outward, only on left
ax.tick_params(axis='y', which='both', direction='out', 
               left=True, right=False, 
               length=6, width=0.8, labelsize=18)

ax.set_ylabel('CDF', fontsize=22)

# ============================================================================
# GRID
# ============================================================================
ax.grid(False)

# ============================================================================
# SAVE & SHOW
# ============================================================================
plt.tight_layout(pad=0.3)
plt.savefig('generic_cr_cdf_quadratic_left.pdf', bbox_inches='tight')
plt.savefig('generic_cr_cdf_quadratic_left.png', bbox_inches='tight', dpi=300)
plt.savefig('generic_cr_cdf_quadratic_left.svg', bbox_inches='tight')

print("\nSaved: generic_cr_cdf_quadratic_left.pdf, .png, .svg")
plt.show()