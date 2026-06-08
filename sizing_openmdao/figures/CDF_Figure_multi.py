import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Set scientific style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150
})

# Create figure and 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))
fig.patch.set_facecolor('white')

# Common x range
x = np.linspace(-2, 2, 1000)

# === 1. Deterministic (top-left) ===
ax = axes[0, 0]
x0 = 0.0
y_cdf = np.where(x < x0, 0.0, 1.0)
ax.plot(x, y_cdf, 'k-', linewidth=2)
ax.plot([x0, x0], [0, 1], 'k-', linewidth=1.5)
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-2, 2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels(['0', '', '1'])
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['', '', '', '', ''])
ax.tick_params(axis='y', labelsize=8)
ax.set_ylabel('CDF', fontsize=9)

# === 2. Aleatory uncertainty (top-right) ===
ax = axes[0, 1]
mean, std = 0.0, 0.5
y_cdf = 0.5 * (1 + erf((x - mean) / (std * np.sqrt(2))))
ax.plot(x, y_cdf, 'k-', linewidth=2)
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-2, 2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels(['0', '', '1'])
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['', '', '', '', ''])
ax.tick_params(axis='y', labelsize=8)

# === 3. Pure interval (bottom-left) with red double-sided arrow ===
ax = axes[1, 0]
lower_bound, upper_bound = -0.8, 0.8
y_cdf_interval = np.where(x < lower_bound, 0.0, 
                           np.where(x < upper_bound, 1.0, 0.0))
ax.plot(x, y_cdf_interval, 'k-', linewidth=2)
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

# Add red double-sided arrow between lower and upper bounds
arrow_y_position = 0.5
ax.annotate('', xy=(lower_bound, arrow_y_position), xytext=(upper_bound, arrow_y_position),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.8))

ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-2, 2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels(['0', '', '1'])
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['', '', '', '', ''])
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_ylabel('CDF', fontsize=9)
ax.set_xlabel('Quantity of interest', fontsize=9)

# === 4. P-box (bottom-right) - Both solid black with red arrow in the middle ===
ax = axes[1, 1]
# Identical shape (same standard deviation), wider shift
std_pbox = 0.3  # Same standard deviation for both
mean_lower = -0.6  # Left CDF (shifted further left)
mean_upper = 0.6   # Right CDF (shifted further right)

lower_cdf = 0.5 * (1 + erf((x - mean_lower) / (std_pbox * np.sqrt(2))))
upper_cdf = 0.5 * (1 + erf((x - mean_upper) / (std_pbox * np.sqrt(2))))

ax.plot(x, lower_cdf, 'k-', linewidth=2)  # Solid black
ax.plot(x, upper_cdf, 'k-', linewidth=2)  # Solid black
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

# Add red double-sided arrow between the two CDFs (at approximately CDF=0.5)
# The arrow should span between the two curves at their midpoint
arrow_y_position_pbox = 0.5
# At y=0.5, find x positions of both CDFs (approximately)
# For a normal CDF with mean=m and std=s, at y=0.5, x = m
# So the arrow goes from mean_lower to mean_upper
ax.annotate('', xy=(mean_lower, arrow_y_position_pbox), xytext=(mean_upper, arrow_y_position_pbox),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.8))

ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-2, 2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels(['0', '', '1'])
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['', '', '', '', ''])
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel('Quantity of interest', fontsize=9)

# Style all spines
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# Save
# fig.savefig('cdf_matrix.pdf', bbox_inches='tight', dpi=300)