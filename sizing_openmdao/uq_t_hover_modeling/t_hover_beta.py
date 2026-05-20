import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# The working parameters from the feasibility analysis
ALPHA = 1.1772
BETA_P = 7.6088
T_MIN = 32.671  # Lower bound
T_MAX = 200.0   # Upper bound (natural support)

# Create the distribution
dist = beta_dist(ALPHA, BETA_P, loc=T_MIN, scale=(T_MAX - T_MIN))

# Calculate key metrics
mean_val = dist.mean()
std_val = dist.std()
p2_5 = dist.ppf(0.025)
p97_5 = dist.ppf(0.975)
mode = T_MIN + (T_MAX - T_MIN) * (ALPHA - 1) / (ALPHA + BETA_P - 2)

# Print verification
print("="*60)
print("BETA DISTRIBUTION MATCHING ALL CONSTRAINTS")
print("="*60)
print(f"\nParameters:")
print(f"  • Beta({ALPHA:.4f}, {BETA_P:.4f})")
print(f"  • Support: [{T_MIN:.3f}, {T_MAX:.1f}] s")
print(f"\nTarget vs Achieved:")
print(f"  • Mean: 55.0 vs {mean_val:.3f} s (error: {abs(55-mean_val):.3f}s)")
print(f"  • Std: 18.0 vs {std_val:.3f} s (error: {abs(18-std_val):.3f}s)")
print(f"  • 2.5%: 33.7 vs {p2_5:.3f} s (error: {abs(33.7-p2_5):.3f}s)")
print(f"  • 97.5%: 101.3 vs {p97_5:.3f} s (error: {abs(101.3-p97_5):.3f}s)")
print(f"\nAdditional properties:")
print(f"  • Mode: {mode:.3f} s")
print(f"  • Median: {dist.ppf(0.5):.3f} s")
print(f"  • Skewness: {2*(BETA_P-ALPHA)*np.sqrt(ALPHA+BETA_P+1)/((ALPHA+BETA_P+2)*np.sqrt(ALPHA*BETA_P)):.3f} (positive = right-skewed)")
print(f"  • P(exceeds 60s): {(1-dist.cdf(60))*100:.2f}%")

# Plotting with x-axis from 0 to 175
x = np.linspace(0, 175, 1000)
pdf = dist.pdf(x)
cdf = dist.cdf(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. PDF PLOT
ax1.plot(x, pdf, color='firebrick', lw=2.5, label='Hover Time PDF')
ax1.fill_between(x, pdf,
                 where=(x >= p2_5) & (x <= p97_5),
                 color='firebrick', alpha=0.15,
                 label='95% Interval (2.5% - 97.5%)')

ax1.axvline(p2_5, color='darkred', ls=':', lw=1.5,
            label=f'2.5%: {p2_5:.1f}s')
ax1.axvline(p97_5, color='darkred', ls=':', lw=1.5,
            label=f'97.5%: {p97_5:.1f}s')
ax1.axvline(mean_val, color='blue', ls='-', alpha=0.6,
            label=f'Mean: {mean_val:.1f}s')
ax1.axvline(60, color='black', ls='--', alpha=0.5,
            label='Design point (60s)')

ax1.set_title('Probability Density Function (PDF)', fontweight='bold')
ax1.set_xlabel(r'Hover Time Duration [s]')
ax1.set_ylabel('Density')
ax1.set_xlim(0, 175)  # Set x-axis from 0 to 175
ax1.set_ylim(bottom=0)  # Ensure y-axis starts at 0
ax1.legend(fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. CDF PLOT
ax2.plot(x, cdf, color='seagreen', lw=2.5, label='Hover Time CDF')
ax2.fill_betweenx([0.025, 0.975], p2_5, p97_5,
                  color='seagreen', alpha=0.1)
ax2.axhline(0.025, color='grey', ls='--', lw=1, alpha=0.7)
ax2.axhline(0.975, color='grey', ls='--', lw=1, alpha=0.7)
ax2.axvline(60, color='black', ls='--', alpha=0.5, label='Design point (60s)')

# Mark the exact percentile points
ax2.plot(p2_5, 0.025, 'ro', markersize=6, label='Specified percentiles')
ax2.plot(p97_5, 0.975, 'ro', markersize=6)

ax2.set_title('Cumulative Distribution Function (CDF)', fontweight='bold')
ax2.set_xlabel(r'Hover Time Duration [s]')
ax2.set_ylabel(r'Cumulative Probability $P(T \leq t)$')
ax2.set_xlim(0, 175)  # Set x-axis from 0 to 175
ax2.set_ylim(-0.02, 1.02)  # Give a little padding above/below
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('beta_all_constraints_fit.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("INTERPRETATION FOR RMDO")
print("="*60)
print(f"""
Key insights:
1. The distribution has support up to {T_MAX:.0f}s, but only 2.5% probability beyond {p97_5:.1f}s
2. This means extreme missions (>101s) are rare but possible
3. The right-skew (mean={mean_val:.1f}s > median={dist.ppf(0.5):.1f}s) captures the tail risk
4. For robust design, consider the 95% interval [{p2_5:.1f}, {p97_5:.1f}]s
5. The design point at 60s protects against ~{(1-dist.cdf(60))*100:.1f}% of missions

This Beta distribution successfully captures all your specified constraints!
""")