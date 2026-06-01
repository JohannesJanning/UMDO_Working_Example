"""
CR Visualization - Journal-style figure matching reference screenshot.
Smooth CDF with proper styling.

Usage:
    python cr_visualization_journal.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================================
# DATA
# ============================================================================

# Your actual values
mean_kg = 7.08
p2_5 = 6.82
p97_5 = 7.76
W_cert = 7.89

# Generate smooth synthetic distribution with more points
np.random.seed(42)
n_samples = 20000  # Increased for smoother CDF

# Create distribution with proper shape
base = np.random.normal(mean_kg, 0.32, n_samples)
# Add slight skew for realism
skew = np.exp(np.random.normal(0, 0.08, n_samples))
W_kg = mean_kg + (base - mean_kg) * skew

# Adjust to match exact percentiles
actual_p97_5 = np.percentile(W_kg, 97.5)
W_kg = W_kg - (actual_p97_5 - p97_5)
actual_mean = np.mean(W_kg)
W_kg = W_kg - (actual_mean - mean_kg)

# Clip to reasonable range
W_kg = W_kg[(W_kg > 5.5) & (W_kg < 9.0)]

# ============================================================================
# SMOOTH CDF USING INTERPOLATION (for publication-quality curve)
# ============================================================================
sorted_W = np.sort(W_kg)
cdf_raw = np.arange(1, len(sorted_W) + 1) / len(sorted_W)

# Create smooth interpolation (1000 points for buttery smooth curve)
n_smooth = 1000
interp_cdf = interp1d(sorted_W, cdf_raw, kind='linear', 
                       fill_value=(0, 1), bounds_error=False)

# Generate smooth x grid
x_smooth = np.linspace(sorted_W.min(), sorted_W.max(), n_smooth)
y_smooth = interp_cdf(x_smooth)

# ============================================================================
# FIND KEY POINTS ON SMOOTH CURVE
# ============================================================================
# Find index where curve passes through mean (50th percentile)
idx_50 = np.argmin(np.abs(y_smooth - 0.50))
w_50 = x_smooth[idx_50]

# Find where curve passes through p97.5 (97.5th percentile)
idx_97_5 = np.argmin(np.abs(y_smooth - 0.975))
w_97_5 = x_smooth[idx_97_5]

# ============================================================================
# COMPUTE CR
# ============================================================================
margin = W_cert - w_97_5
uncertainty = w_97_5 - mean_kg
cr = margin / uncertainty if uncertainty > 0 else np.nan

# ============================================================================
# MATPLOTLIB STYLE (matching reference screenshot)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.1,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False
})

# ============================================================================
# FIGURE (matching the screenshot dimensions)
# ============================================================================
fig, ax = plt.subplots(figsize=(5.2, 3.5))

# ============================================================================
# SMOOTH CDF CURVE
# ============================================================================
ax.plot(x_smooth, y_smooth, color='black', lw=1.2, zorder=3)

# Anchor dot on the curve at the P2.5 boundary (matching reference style)
idx_2_5 = np.argmin(np.abs(x_smooth - p2_5))
ax.plot(p2_5, y_smooth[idx_2_5], 'o', color='black', markersize=2.5, zorder=4)

# ============================================================================
# PROBABILITY REQUIREMENT LINE (dashed horizontal)
# ============================================================================
prob_req = 0.975
ax.axhline(y=prob_req, color='black', linestyle='--', lw=0.7, zorder=2)

# "Probability Requirement" label
ax.text(6.22, prob_req - 0.02, "Probability\nRequirement", 
        va='top', ha='left', fontsize=7.5, style='italic')

# ============================================================================
# VERTICAL LINES (dashed/solid, going down to x-axis or up to markers)
# ============================================================================
# Line at P2.5 - goes up to the curve anchor point
ax.vlines(x=p2_5, ymin=0, ymax=y_smooth[idx_2_5], color='black', linestyle='--', lw=0.7, zorder=2)

# Line at μ (mean) - solid line spanning all the way from bottom to top frame
ax.axvline(x=mean_kg, color='black', linestyle='-', lw=0.7, zorder=2)

# Line at P97.5 - goes down to prob_req line
ax.vlines(x=w_97_5, ymin=0, ymax=prob_req, color='black', linestyle='--', lw=0.7, zorder=2)

# ============================================================================
# W_cert LINE & FAILURE BOUNDARY (red, as in screenshot)
# ============================================================================
ax.axvline(x=W_cert, ymin=0, ymax=prob_req, color='#991B1B', lw=1.2, zorder=2)

# Horizontal continuation of red line into failure region
ax.plot([W_cert, 8.2], [prob_req, prob_req], color='#991B1B', lw=1.2, zorder=2)

# "Failure Region" label
ax.text(8.05, 0.55, "Failure\nRegion", color='#991B1B', 
        rotation=90, ha='center', va='center', fontsize=9, weight='bold')

# ============================================================================
# ANNOTATION ARROWS FOR U AND M
# ============================================================================
arrow_style = dict(arrowstyle='<->', color='black', lw=0.6, shrinkA=0, shrinkB=0)

# U arrow (uncertainty) - horizontal between μ and P97.5
y_arrow_u = 0.25
ax.annotate('', xy=(mean_kg, y_arrow_u), xytext=(w_97_5, y_arrow_u), arrowprops=arrow_style)
ax.text((mean_kg + w_97_5)/2, y_arrow_u + 0.02, r'$\mathrm{Uncertainty}\ (U)$', 
        ha='center', va='bottom', fontsize=8)

# M arrow (margin) - horizontal between P97.5 and W_cert
y_arrow_m = 0.45
ax.annotate('', xy=(w_97_5, y_arrow_m), xytext=(W_cert, y_arrow_m), arrowprops=arrow_style)
ax.text((w_97_5 + W_cert)/2, y_arrow_m + 0.02, r'$\mathrm{Margin}\ (M)$', 
        ha='center', va='bottom', fontsize=8)

# ============================================================================
# X-AXIS: Clean number indicators
# ============================================================================
x_ticks = [6.2, 6.6, 7.0, 7.4, 7.8, 8.2]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
ax.set_xlabel(r'$\mathrm{W}_{\mathrm{total}}\ [\mathrm{kg}]$', labelpad=6)

# ============================================================================
# Y-AXIS
# ============================================================================
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_ylabel(r'$\mathrm{CDF}$')

ax.set_xlim(6.2, 8.2)

# ============================================================================
# TICKS & SPINES (enclosed box with inward ticks)
# ============================================================================
ax.tick_params(direction='in', top=False, right=True, length=2.5, width=0.5)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.grid(False)

# ============================================================================
# TWIN TOP AXIS FOR SMALLER INDICATORS
# ============================================================================
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks([p2_5, mean_kg, w_97_5, W_cert])
# Making the text labels smaller as requested (fontsize=7.5)
ax_top.set_xticklabels([r'$P_{2.5}$', r'$\mu$', r'$P_{97.5}$', r'$W_{\mathrm{cert}}$'], fontsize=7.5)
ax_top.tick_params(direction='in', width=0.5, length=3.0)

# ============================================================================
# SAVE & SHOW
# ============================================================================
plt.tight_layout()
plt.savefig('cr_cdf_journal.pdf', bbox_inches='tight')
plt.savefig('cr_cdf_journal.png', bbox_inches='tight', dpi=300)
plt.savefig('cr_cdf_journal.svg', bbox_inches='tight')

plt.show()