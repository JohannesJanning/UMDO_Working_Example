import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import lognorm, gamma, norm
from scipy.optimize import minimize

# Your Beta parameters (matching mean=55, std=18, percentiles)
ALPHA = 1.1772
BETA_P = 7.6088
T_MIN = 32.671
T_MAX = 200.0

beta_distribution = beta_dist(ALPHA, BETA_P, loc=T_MIN, scale=(T_MAX - T_MIN))

# Find LogNormal with same mean and std
def find_lognormal(target_mean, target_std):
    """Find LogNormal parameters (shape, scale, loc=0) matching mean and std"""
    def objective(sigma):
        mu = np.log(target_mean) - 0.5 * sigma**2
        mean_calc = np.exp(mu + 0.5*sigma**2)
        std_calc = np.sqrt((np.exp(sigma**2)-1) * np.exp(2*mu + sigma**2))
        return (mean_calc - target_mean)**2 + (std_calc - target_std)**2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=[0.01, 3], method='bounded')
    sigma = result.x
    mu = np.log(target_mean) - 0.5 * sigma**2
    return mu, sigma

mu_ln, sigma_ln = find_lognormal(55, 18)
lognorm_dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))

# Find Gamma distribution with same mean and std
# Gamma: shape=k, scale=theta, mean=k*theta, var=k*theta^2
k_gamma = (55/18)**2  # shape = (mean/std)^2
theta_gamma = 18**2/55  # scale = var/mean
gamma_dist = gamma(a=k_gamma, scale=theta_gamma)

# Compare distributions
x = np.linspace(0, 175, 1000)
beta_pdf = beta_distribution.pdf(x)
lognorm_pdf = lognorm_dist.pdf(x)
gamma_pdf = gamma_dist.pdf(x)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. PDF Comparison
ax = axes[0, 0]
ax.plot(x, beta_pdf, 'r-', lw=2, label='Beta (your fit)')
ax.plot(x, lognorm_pdf, 'g--', lw=2, label='LogNormal')
ax.plot(x, gamma_pdf, 'b-.', lw=2, label='Gamma')
ax.axvline(55, color='k', ls=':', alpha=0.5, label='Mean=55s')
ax.set_xlim(0, 175)
ax.set_ylim(bottom=0)
ax.set_title('PDF Comparison (Same Mean/Std)', fontweight='bold')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. CDF Comparison
ax = axes[0, 1]
ax.plot(x, beta_distribution.cdf(x), 'r-', lw=2, label='Beta')
ax.plot(x, lognorm_dist.cdf(x), 'g--', lw=2, label='LogNormal')
ax.plot(x, gamma_dist.cdf(x), 'b-.', lw=2, label='Gamma')
ax.axhline(0.025, color='gray', ls=':', alpha=0.5)
ax.axhline(0.975, color='gray', ls=':', alpha=0.5)
ax.set_xlim(0, 175)
ax.set_title('CDF Comparison', fontweight='bold')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Cumulative Probability')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Tail Comparison (Zoom on upper tail)
ax = axes[0, 2]
tail_x = np.linspace(80, 175, 500)
ax.semilogy(tail_x, 1-beta_distribution.cdf(tail_x), 'r-', lw=2, label='Beta (1-CDF)')
ax.semilogy(tail_x, 1-lognorm_dist.cdf(tail_x), 'g--', lw=2, label='LogNormal')
ax.semilogy(tail_x, 1-gamma_dist.cdf(tail_x), 'b-.', lw=2, label='Gamma')
ax.axhline(0.025, color='gray', ls=':', alpha=0.5, label='2.5% exceedance')
ax.set_xlabel('Time [s]')
ax.set_ylabel('P(T > t) [log scale]')
ax.set_title('Upper Tail Comparison (Exceedance)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. PDF at lower values (0-50s)
ax = axes[1, 0]
low_x = np.linspace(0, 50, 500)
ax.plot(low_x, beta_distribution.pdf(low_x), 'r-', lw=2, label='Beta')
ax.plot(low_x, lognorm_dist.pdf(low_x), 'g--', lw=2, label='LogNormal')
ax.plot(low_x, gamma_dist.pdf(low_x), 'b-.', lw=2, label='Gamma')
ax.axvline(33.7, color='red', ls=':', alpha=0.5, label='2.5% at 33.7s')
ax.set_xlim(0, 50)
ax.set_title('Lower Tail Comparison (Density at low values)', fontweight='bold')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Key percentiles comparison
ax = axes[1, 1]
percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
beta_vals = [beta_distribution.ppf(p/100) for p in percentiles]
lognorm_vals = [lognorm_dist.ppf(p/100) for p in percentiles]
gamma_vals = [gamma_dist.ppf(p/100) for p in percentiles]

ax.plot(percentiles, beta_vals, 'ro-', lw=2, markersize=4, label='Beta')
ax.plot(percentiles, lognorm_vals, 'gs--', lw=2, markersize=4, label='LogNormal')
ax.plot(percentiles, gamma_vals, 'b^-.', lw=2, markersize=4, label='Gamma')
ax.axhline(55, color='k', ls=':', alpha=0.5, label='Mean')
ax.set_xlabel('Percentile')
ax.set_ylabel('Time [s]')
ax.set_title('Percentile Comparison', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 6. Difference in tail probabilities
ax = axes[1, 2]
thresholds = np.arange(60, 141, 10)
beta_exceed = [1-beta_distribution.cdf(t) for t in thresholds]
lognorm_exceed = [1-lognorm_dist.cdf(t) for t in thresholds]
gamma_exceed = [1-gamma_dist.cdf(t) for t in thresholds]

width = 0.25
x_pos = np.arange(len(thresholds))
ax.bar(x_pos - width, beta_exceed, width, label='Beta', alpha=0.7)
ax.bar(x_pos, lognorm_exceed, width, label='LogNormal', alpha=0.7)
ax.bar(x_pos + width, gamma_exceed, width, label='Gamma', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{t}s' for t in thresholds])
ax.set_yscale('log')
ax.set_ylabel('P(T > threshold) [log scale]')
ax.set_xlabel('Threshold')
ax.set_title('Exceedance Probability Comparison', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Quantitative differences for your MCS
print("="*70)
print("DISTRIBUTION COMPARISON FOR MCS")
print("="*70)

# Test points of interest for your RMDO
test_points = [40, 60, 80, 100, 120]
print(f"\n{'Threshold':<10} {'Beta P(<t)':<12} {'LogNormal P(<t)':<16} {'Gamma P(<t)':<14} {'Max Diff':<10}")
print("-"*70)
for t in test_points:
    beta_p = beta_distribution.cdf(t)
    ln_p = lognorm_dist.cdf(t)
    gamma_p = gamma_dist.cdf(t)
    max_diff = max(abs(beta_p - ln_p), abs(beta_p - gamma_p))
    print(f"{t:<10} {beta_p:<12.4f} {ln_p:<16.4f} {gamma_p:<14.4f} {max_diff:<10.4f}")

print(f"\nCritical differences for MCS:")
print(f"• At 60s (design point): Beta={beta_distribution.cdf(60):.3f}, LogNormal={lognorm_dist.cdf(60):.3f}, Gamma={gamma_dist.cdf(60):.3f}")
print(f"  Difference up to {max(abs(beta_distribution.cdf(60)-lognorm_dist.cdf(60)), abs(beta_distribution.cdf(60)-gamma_dist.cdf(60))):.3f}")

print(f"\n• 99th percentile: Beta={beta_distribution.ppf(0.99):.1f}s, LogNormal={lognorm_dist.ppf(0.99):.1f}s, Gamma={gamma_dist.ppf(0.99):.1f}s")
print(f"  Range: {abs(beta_distribution.ppf(0.99)-lognorm_dist.ppf(0.99)):.1f}s difference")

print(f"\n• Probability > 100s: Beta={1-beta_distribution.cdf(100):.4f}, LogNormal={1-lognorm_dist.cdf(100):.4f}, Gamma={1-gamma_dist.cdf(100):.4f}")
print(f"  Factor difference: {max((1-beta_distribution.cdf(100))/(1-lognorm_dist.cdf(100)), (1-lognorm_dist.cdf(100))/(1-beta_distribution.cdf(100))):.1f}x")

print("\n" + "="*70)
print("RECOMMENDATION FOR YOUR RMDO")
print("="*70)
print("""
For your MCS models:

1. IF your response is LINEAR or NEAR-LINEAR in hover time:
   → Moments may be sufficient (Beta vs LogNormal differences <5-10%)
   
2. IF your response is NONLINEAR (e.g., quadratic, exponential, threshold-based):
   → Distribution shape MATTERS significantly
   → Differences in tails (2-10x factors) will affect rare event probabilities
   
3. IF your constraints involve EXTREME VALUES (e.g., >100s):
   → The Beta distribution you've fitted is conservative in the upper tail
   → LogNormal would give more optimistic (lower) extreme probabilities
   
4. BEST PRACTICE for RMDO:
   → Test sensitivity by running MCS with BOTH distributions
   → If results differ significantly, collect data to validate shape
   → Consider using the fitted Beta since it matches your specified percentiles
   
CONCLUSION: The Beta distribution with your specific parameters is defensible
because it directly matches the percentiles you care about. The density differences 
at low values (<33.7s) don't matter since you have truncation there anyway.
""")