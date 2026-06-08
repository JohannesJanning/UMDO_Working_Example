import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# NORMALIZED GAUSSIAN DISTRIBUTION (Standard Normal)
# ============================================================================

# Parameters for standard normal distribution
mean = 0.0
std = 1.0

# Generate x values and PDF
x = np.linspace(-4, 4, 1000)
y_pdf = norm.pdf(x, loc=mean, scale=std)

# Calculate 2.5% and 97.5% percentiles (95% confidence interval)
p2_5 = norm.ppf(0.025, loc=mean, scale=std)   # -1.96
p97_5 = norm.ppf(0.975, loc=mean, scale=std)  # 1.96

# ============================================================================
# MATPLOTLIB STYLE (clean, no numbering)
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
fig, ax = plt.subplots(figsize=(5.2, 5.2))  # Square figure

# PDF curve (black)
ax.plot(x, y_pdf, color='black', lw=1.5)

# ============================================================================
# CONFIDENCE INTERVAL DASHED LINES
# ============================================================================
# Vertical dashed lines at 2.5% and 97.5% (RED)
ax.axvline(x=p2_5, color='red', linestyle='--', lw=1.0, alpha=0.7)
ax.axvline(x=p97_5, color='red', linestyle='--', lw=1.0, alpha=0.7)

# Shade the 95% confidence interval area (between p2_5 and p97_5)
x_fill = np.linspace(p2_5, p97_5, 500)
y_fill = norm.pdf(x_fill, loc=mean, scale=std)
ax.fill_between(x_fill, y_fill, alpha=0.15, color='gray')

# ============================================================================
# X-AXIS (no numbers, only major ticks)
# ============================================================================
ax.set_xlim(-4, 4)

# Set major ticks but no labels
ax.set_xticks(np.arange(-4, 4.5, 1))
ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])  # No numbers

# No minor ticks
ax.set_xticks([], minor=True)

# Keep all spines (full frame/box)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Ticks pointing outward, only on bottom
ax.tick_params(axis='x', which='major', direction='out', 
               top=False, bottom=True, 
               length=6, width=0.8, labelsize=18)

ax.set_xlabel('Response value', fontsize=22)

# ============================================================================
# Y-AXIS (no numbers, only major ticks)
# ============================================================================
ax.set_ylim(0, 0.45)

# Set major ticks but no labels
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(['', '', '', '', ''])  # No numbers

# No minor ticks
ax.set_yticks([], minor=True)

# Ticks pointing outward, only on left
ax.tick_params(axis='y', which='major', direction='out', 
               left=True, right=False, 
               length=6, width=0.8, labelsize=18)

ax.set_ylabel('PDF', fontsize=22)

# ============================================================================
# GRID (optional, light gray for reference)
# ============================================================================
ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

# ============================================================================
# SAVE & SHOW
# ============================================================================
plt.tight_layout(pad=0.3)
plt.savefig('gaussian_pdf_ci_clean.pdf', bbox_inches='tight')
plt.savefig('gaussian_pdf_ci_clean.png', bbox_inches='tight', dpi=300)
plt.savefig('gaussian_pdf_ci_clean.svg', bbox_inches='tight')

print("\nSaved: gaussian_pdf_ci_clean.pdf, .png, .svg")
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# NORMALIZED GAUSSIAN DISTRIBUTION (Standard Normal) - CDF
# ============================================================================

# Parameters for standard normal distribution
mean = 0.0
std = 1.0

# Generate x values and CDF
x = np.linspace(-4, 4, 1000)
y_cdf = norm.cdf(x, loc=mean, scale=std)

# Calculate 2.5% and 97.5% percentiles (95% confidence interval)
p2_5 = norm.ppf(0.025, loc=mean, scale=std)   # -1.96
p97_5 = norm.ppf(0.975, loc=mean, scale=std)  # 1.96

# CDF values at the percentiles
cdf_at_p2_5 = 0.025
cdf_at_p97_5 = 0.975

# ============================================================================
# MATPLOTLIB STYLE (clean, no numbering)
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
fig, ax = plt.subplots(figsize=(5.2, 5.2))  # Square figure

# CDF curve (black)
ax.plot(x, y_cdf, color='black', lw=1.5)

# ============================================================================
# CONFIDENCE INTERVAL DASHED LINES
# ============================================================================
# Vertical dashed lines at 2.5% and 97.5% percentiles (RED)
ax.axvline(x=p2_5, color='red', linestyle='--', lw=1.0, alpha=0.7)
ax.axvline(x=p97_5, color='red', linestyle='--', lw=1.0, alpha=0.7)

# Horizontal dashed lines at the corresponding CDF values (BLACK)
ax.axhline(y=cdf_at_p2_5, color='black', linestyle='--', lw=1.0, alpha=0.7)
ax.axhline(y=cdf_at_p97_5, color='black', linestyle='--', lw=1.0, alpha=0.7)

# ============================================================================
# X-AXIS (no numbers, only major ticks)
# ============================================================================
ax.set_xlim(-4, 4)

# Set major ticks but no labels
ax.set_xticks(np.arange(-4, 4.5, 1))
ax.set_xticklabels(['', '', '', '', '', '', '', '', ''])  # No numbers

# No minor ticks
ax.set_xticks([], minor=True)

# Keep all spines (full frame/box)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Ticks pointing outward, only on bottom
ax.tick_params(axis='x', which='major', direction='out', 
               top=False, bottom=True, 
               length=6, width=0.8, labelsize=18)

ax.set_xlabel('Response value', fontsize=22)

# ============================================================================
# Y-AXIS (only goes to 1, no numbers, only major ticks)
# ============================================================================
ax.set_ylim(0, 1.0)

# Set major ticks but no labels
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['', '', '', '', '', ''])  # No numbers

# No minor ticks
ax.set_yticks([], minor=True)

# Ticks pointing outward, only on left
ax.tick_params(axis='y', which='major', direction='out', 
               left=True, right=False, 
               length=6, width=0.8, labelsize=18)

ax.set_ylabel('CDF', fontsize=22)

# ============================================================================
# GRID (optional, light gray for reference)
# ============================================================================
ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

# ============================================================================
# SAVE & SHOW
# ============================================================================
plt.tight_layout(pad=0.3)
plt.savefig('gaussian_cdf_ci_clean.pdf', bbox_inches='tight')
plt.savefig('gaussian_cdf_ci_clean.png', bbox_inches='tight', dpi=300)
plt.savefig('gaussian_cdf_ci_clean.svg', bbox_inches='tight')

print("\nSaved: gaussian_cdf_ci_clean.pdf, .png, .svg")
plt.show()