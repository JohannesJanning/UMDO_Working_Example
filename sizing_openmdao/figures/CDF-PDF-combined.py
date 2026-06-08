"""
Unified PDF Visualization - Three Design Strategies
Single figure with x-axis limited to 8.2 kg for clear comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================================
# FUNCTION TO GENERATE MCS SAMPLES FOR A GIVEN DESIGN
# ============================================================================
def generate_samples_for_design(x_det, design_name, n_mc=10000, seed_offset=0):
    """Generate W_total samples for a given deterministic design"""
    
    from run_qbit_MCS import RobustOptimizer, inner_solve_for_Wtotal, sample_t_hover
    
    uq = RobustOptimizer(payload_kg=3.0, range_m=15000.0, n_c=2,
                         n_mc=n_mc, seed=123 + seed_offset)
    
    samples = sample_t_hover(n_mc, uq.mean_t, uq.std_t, uq.shift_t, seed=123 + seed_offset)
    
    W_vals = []
    for t in samples:
        res = inner_solve_for_Wtotal(t, uq.payload_kg, uq.range_m, uq.n_c, 
                                      design_vars=tuple(x_det))
        if isinstance(res, dict):
            W_vals.append(res['W_total'])
    
    W_kg = np.array(W_vals) / 9.80665
    
    # Calculate statistics
    mean_kg = np.mean(W_kg)
    p97_5 = np.percentile(W_kg, 97.5)
    p2_5 = np.percentile(W_kg, 2.5)
    std_kg = np.std(W_kg)
    
    print(f"{design_name:12s}: μ={mean_kg:.3f} kg, σ={std_kg:.3f} kg, P97.5={p97_5:.3f} kg")
    
    return W_kg, mean_kg, p97_5, p2_5, std_kg

# ============================================================================
# DEFINE THREE DESIGNS
# ============================================================================
designs = {
    'Conservative': [29.33, 0.2698, 1.3, 0.2435],
    'Nominal':      [31.36, 0.2227, 1.3, 0.1895],
    'MRDO':         [28.98, 0.2622, 1.3, 0.2245]
}

# Generate samples for each design
samples = {}
stats = {}

for idx, (name, x_det) in enumerate(designs.items()):
    W_samples, mean_val, p97_5_val, p2_5_val, std_val = generate_samples_for_design(
        x_det, name, n_mc=10000, seed_offset=idx
    )
    samples[name] = W_samples
    stats[name] = {'mean': mean_val, 'p97.5': p97_5_val, 'p2.5': p2_5_val, 'std': std_val}

# ============================================================================
# CREATE PDF USING KERNEL DENSITY ESTIMATION
# ============================================================================
# Common x-range (cut at 8.2 kg)
x_max_display = 8.2
x_min_display = 6.6
x_range = np.linspace(x_min_display, x_max_display, 1000)

pdfs = {}
for name, W_data in samples.items():
    # Filter samples to x_range for better KDE
    W_filtered = W_data[W_data <= x_max_display]
    if len(W_filtered) < 100:
        W_filtered = W_data
    
    kde = gaussian_kde(W_filtered, bw_method='scott')
    pdf_vals = kde(x_range)
    pdfs[name] = pdf_vals

# Find y_max for consistent scaling (excluding extreme tails)
y_max = max([np.max(pdf) for pdf in pdfs.values()]) * 1.15

# ============================================================================
# W_cert value (from deterministic analysis)
# ============================================================================
W_cert = 7.854  # kg

# ============================================================================
# MATPLOTLIB STYLE - SCIENTIFIC JOURNAL FORMAT
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False,
    'figure.figsize': (5.5, 4.0)
})

# ============================================================================
# CREATE FIGURE - SINGLE PANEL
# ============================================================================
fig, ax = plt.subplots()

# Colors (colorblind-friendly + professional)
colors = {'Conservative': '#377eb8', 'Nominal': '#e41a1c', 'MRDO': '#4daf4a'}
linestyles = {'Conservative': '-', 'Nominal': '--', 'MRDO': '-.'}
linewidths = {'Conservative': 1.5, 'Nominal': 1.5, 'MRDO': 1.5}

# Plot PDFs
for name in designs.keys():
    ax.plot(x_range, pdfs[name], 
            color=colors[name], 
            linestyle=linestyles[name],
            linewidth=linewidths[name], 
            label=f'{name}')

# Add vertical line for W_cert
ax.axvline(x=W_cert, color='black', linestyle='-', linewidth=1.0, alpha=0.8, 
           label=f'$W_{{cert}}$ = {W_cert} kg')

# Add mean markers (small triangles)
for name in designs.keys():
    if x_min_display <= stats[name]['mean'] <= x_max_display:
        ax.plot(stats[name]['mean'], 0.02, 'v', 
                markersize=5, color=colors[name], 
                markeredgecolor='black', markeredgewidth=0.3, clip_on=False)

# Add P97.5 markers (small inverted triangles)
for name in designs.keys():
    if x_min_display <= stats[name]['p97.5'] <= x_max_display:
        ax.plot(stats[name]['p97.5'], 0.02, '^', 
                markersize=5, color=colors[name],
                markeredgecolor='black', markeredgewidth=0.3, clip_on=False)

# Shade failure region (W > W_cert)
failure_mask = x_range > W_cert
if np.any(failure_mask):
    # Light red shading for failure region
    max_pdf_in_failure = max([np.max(pdfs[name][failure_mask]) for name in designs.keys()])
    ax.fill_between(x_range[failure_mask], 0, max_pdf_in_failure * 1.05,
                    alpha=0.08, color='red', zorder=0)
    
    # Add "Failure Region" label
    ax.text(W_cert + 0.12, max_pdf_in_failure * 0.6, 'Failure\nRegion',
            color='red', fontsize=9, ha='left', va='center', style='italic')

# ============================================================================
# AXIS FORMATTING
# ============================================================================
# X-axis
ax.set_xlim(x_min_display, x_max_display)
ax.set_xlabel('$W_{total}$ [kg]', fontsize=11)

# Major ticks every 0.2 kg
major_ticks = np.arange(6.6, 8.3, 0.2)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'{tick:.1f}' for tick in major_ticks])

# Minor ticks every 0.1 kg
minor_ticks = np.arange(6.6, 8.3, 0.1)
ax.set_xticks(minor_ticks, minor=True)

# Y-axis
ax.set_ylim(0, y_max)
ax.set_ylabel('Probability density', fontsize=11)

# Scientific y-tick formatting (optional: use 1.5, 1.0, 0.5 instead of decimals)
ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

# ============================================================================
# TICKS AND SPINES
# ============================================================================
ax.tick_params(direction='in', top=True, right=True, 
               length=4, width=0.6, labelsize=10)
ax.tick_params(which='minor', direction='in', top=True, right=True,
               length=2, width=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_visible(True)

# ============================================================================
# LEGEND
# ============================================================================
legend = ax.legend(loc='upper right', frameon=False, fontsize=10, handlelength=2.5)
for line in legend.get_lines():
    line.set_linewidth(1.5)

# ============================================================================
# INSET ANNOTATION FOR MARKERS (optional)
# ============================================================================
# Add small annotation explaining the markers
ax.text(0.98, 0.95, '▼ Mean\n▲ P97.5', transform=ax.transAxes,
        fontsize=8, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

# ============================================================================
# SAVE FIGURE
# ============================================================================
plt.tight_layout(pad=0.3)
plt.savefig('pdf_comparison_unified.pdf', bbox_inches='tight')
plt.savefig('pdf_comparison_unified.png', bbox_inches='tight', dpi=300)
plt.savefig('pdf_comparison_unified.svg', bbox_inches='tight')

# ============================================================================
# PRINT STATISTICS
# ============================================================================
print("\n" + "="*70)
print("DESIGN COMPARISON SUMMARY (x-axis cut at 8.2 kg)")
print("="*70)
print(f"{'Design':<12} {'Mean [kg]':<12} {'P97.5 [kg]':<12} {'P2.5 [kg]':<12} {'P(W>W_cert)':<12}")
print("-"*70)
for name in designs.keys():
    failure_prob = np.sum(samples[name] > W_cert) / len(samples[name])
    print(f"{name:<12} {stats[name]['mean']:<12.3f} {stats[name]['p97.5']:<12.3f} "
          f"{stats[name]['p2.5']:<12.3f} {failure_prob:<12.4f}")
print("="*70)

plt.show()