"""
CR Visualization - Using actual MCS samples from UQ evaluation.

This script loads the W_total samples from your UQ evaluation results
and creates the CDF figure directly from the real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================================
# LOAD OR GENERATE MCS SAMPLES
# ============================================================================

try:
    W_kg = np.load('mtom_samples_umdo.npy')
    print(f"Loaded {len(W_kg)} MCS samples from file")
except FileNotFoundError:
    print("No saved samples found. Using design point to generate samples...")
    
    x_det = [29.33, 0.2698, 1.3, 0.2435]
    
    from run_qbit_MCS import RobustOptimizer, inner_solve_for_Wtotal, sample_t_hover
    
    n_mc = 1000
    uq = RobustOptimizer(payload_kg=3.0, range_m=15000.0, n_c=2,
                         n_mc=n_mc, seed=123)
    
    samples = sample_t_hover(n_mc, uq.mean_t, uq.std_t, uq.shift_t, seed=123)
    
    W_vals = []
    for t in samples:
        res = inner_solve_for_Wtotal(t, uq.payload_kg, uq.range_m, uq.n_c, 
                                      design_vars=tuple(x_det))
        if isinstance(res, dict):
            W_vals.append(res['W_total'])
    
    W_kg = np.array(W_vals) / 9.80665
    print(f"Generated {len(W_kg)} MCS samples")

# ============================================================================
# CALCULATE STATISTICS FROM ACTUAL DATA
# ============================================================================
mean_kg = float(np.mean(W_kg))
p2_5 = float(np.percentile(W_kg, 2.5))
p97_5 = float(np.percentile(W_kg, 97.5))
p50 = float(np.percentile(W_kg, 50.0))

# W_cert from your results (nominal deterministic upper bound)
W_cert = 7.854

# Find the CDF value at each point (for drawing the vertical lines and dots)
sorted_W = np.sort(W_kg)
cdf_raw = np.arange(1, len(sorted_W) + 1) / len(sorted_W)

# Create interpolation functions
interp_cdf = interp1d(sorted_W, cdf_raw, kind='linear', 
                       fill_value=(0, 1), bounds_error=False)
interp_cdf_for_point = interp1d(sorted_W, cdf_raw, kind='linear', 
                                 fill_value=(0, 1), bounds_error=False)

# Get CDF values at the key points
cdf_at_mean = float(interp_cdf_for_point(mean_kg))
cdf_at_p2_5 = float(interp_cdf_for_point(p2_5))
cdf_at_p97_5 = float(interp_cdf_for_point(p97_5))

print("\n" + "="*50)
print("Statistics from MCS samples:")
print("="*50)
print(f"  Number of samples: {len(W_kg)}")
print(f"  μ (mean)         : {mean_kg:.3f} kg (CDF: {cdf_at_mean:.3f})")
print(f"  P₂.₅ (2.5%)      : {p2_5:.3f} kg (CDF: {cdf_at_p2_5:.3f})")
print(f"  P₅₀ (median)     : {p50:.3f} kg")
print(f"  P₉₇.₅ (97.5%)    : {p97_5:.3f} kg (CDF: {cdf_at_p97_5:.3f})")
print(f"  W_cert           : {W_cert:.3f} kg")

# ============================================================================
# COMPUTE CR
# ============================================================================
margin = W_cert - p97_5
uncertainty = p97_5 - mean_kg
cr = margin / uncertainty if uncertainty > 0 else np.nan

print(f"\n  Margin (M)       : {margin*1000:.1f} g")
print(f"  Uncertainty (U)  : {uncertainty*1000:.1f} g")
print(f"  CR = M/U         : {cr:.4f}")
print("="*50)

# ============================================================================
# CREATE SMOOTH CDF
# ============================================================================
n_smooth = 1000
interp_cdf_smooth = interp1d(sorted_W, cdf_raw, kind='linear', 
                              fill_value=(0, 1), bounds_error=False)
x_smooth = np.linspace(6.6, 8.1, n_smooth)
y_smooth = interp_cdf_smooth(x_smooth)

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 20,                    # change this
    'axes.labelsize': 20,               # change this
    'axes.titlesize': 20,
    'xtick.labelsize': 20,              # change this
    'ytick.labelsize': 20,              # change this
    'legend.fontsize': 20,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False
})

# ============================================================================
# FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(5.2, 4.0))

# CDF curve from actual data
ax.plot(x_smooth, y_smooth, color='black', lw=1.2)

# ============================================================================
# DOTS AT INTERSECTION POINTS (on the CDF curve)
# ============================================================================
ax.plot(p2_5, cdf_at_p2_5, 'ko', markersize=4, markeredgewidth=0.5)
ax.plot(mean_kg, cdf_at_mean, 'ko', markersize=4, markeredgewidth=0.5)
ax.plot(p97_5, cdf_at_p97_5, 'ko', markersize=4, markeredgewidth=0.5)

# ============================================================================
# PROBABILITY REQUIREMENT LINES
# ============================================================================
prob_req_high = 0.975
ax.axhline(y=prob_req_high, color='black', linestyle='--', lw=0.8)
ax.text(6.65, prob_req_high - 0.02, "Probability\nRequirement", 
        va='top', ha='left', fontsize=16)

prob_req_low = 0.025
ax.axhline(y=prob_req_low, color='black', linestyle='--', lw=0.8)

# ============================================================================
# VERTICAL LINES
# ============================================================================
ax.vlines(x=mean_kg, ymin=0, ymax=cdf_at_mean, color='black', linestyle='--', lw=0.8)
ax.vlines(x=p2_5, ymin=0, ymax=prob_req_low, color='black', linestyle='--', lw=0.8)
ax.vlines(x=p97_5, ymin=0, ymax=prob_req_high, color='black', linestyle='--', lw=0.8)

# ============================================================================
# W_cert LINE (red) AND CONDITIONAL MARGIN COLORING
# ============================================================================
ax.axvline(x=W_cert, ymin=0, ymax=prob_req_high, color='red', lw=1.0)
ax.plot([W_cert, 8.1], [prob_req_high, prob_req_high], color='red', lw=1.0)

# Conditional shading for the margin area ONLY (between P97.5 and W_cert)
if margin > 0:
    # Positive margin - light green shading
    ax.axvspan(p97_5, W_cert, ymin=0, ymax=prob_req_high, 
               alpha=0.15, color='green')
    margin_color = 'green'
elif margin < 0:
    # Negative margin - light red shading (deficit)
    ax.axvspan(W_cert, p97_5, ymin=0, ymax=prob_req_high, 
               alpha=0.15, color='red')
    margin_color = 'red'
else:
    margin_color = 'black'

# ============================================================================
# FAILURE REGION LABEL (always says "Failure Region", always red)
# ============================================================================
failure_center_x = (W_cert + 8.1) / 2
failure_center_y = prob_req_high / 2
ax.text(failure_center_x, failure_center_y, "Failure Region", color='red', 
        rotation=90, ha='center', va='center', fontsize=16)

# ============================================================================
# ARROWS
# ============================================================================
# U arrow (uncertainty)
y_arrow_u = 0.25
ax.annotate('', xy=(mean_kg, y_arrow_u), xytext=(p97_5, y_arrow_u),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.7))
ax.text((mean_kg + p97_5)/2, y_arrow_u + 0.02, 'U', 
        ha='center', va='bottom', fontsize=16, fontstyle='italic')

# M arrow (margin) - color-coded to match margin (green/red)
y_arrow_m = 0.45
ax.annotate('', xy=(p97_5, y_arrow_m), xytext=(W_cert, y_arrow_m),
            arrowprops=dict(arrowstyle='<->', color=margin_color, lw=0.7))
ax.text((p97_5 + W_cert)/2, y_arrow_m + 0.02, 'M', 
        ha='center', va='bottom', fontsize=16, fontstyle='italic', color=margin_color)

# ============================================================================
# X-AXIS
# ============================================================================
ax.set_xlim(6.6, 8.1)

major_ticks = np.arange(6.6, 8.2, 0.2)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'{tick:.1f}' for tick in major_ticks])

minor_ticks = np.arange(6.6, 8.2, 0.1)
ax.set_xticks(minor_ticks, minor=True)

ax.set_xlabel('W_total [kg]', fontsize=16)

# ============================================================================
# Y-AXIS
# ============================================================================
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_ylabel('CDF', fontsize=16)

# ============================================================================
# TICKS & SPINES
# ============================================================================
ax.tick_params(direction='in', top=True, right=True, 
               length=4, width=0.6, labelsize=16)

ax.tick_params(which='minor', direction='in', top=True, right=True,
               length=2, width=0.4)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.grid(False)

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout(pad=0.3)
plt.savefig('cr_cdf_journal.pdf', bbox_inches='tight')
plt.savefig('cr_cdf_journal.png', bbox_inches='tight', dpi=300)
plt.savefig('cr_cdf_journal.svg', bbox_inches='tight')

print("\nSaved: cr_cdf_journal.pdf, .png, .svg")
plt.show()