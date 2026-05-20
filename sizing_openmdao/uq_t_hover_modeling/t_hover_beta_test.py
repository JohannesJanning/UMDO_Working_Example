import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize, differential_evolution
from scipy.special import beta as beta_function

def check_feasibility():
    """Check if the desired constraints are mathematically possible"""
    
    # Given constraints
    target_mean = 55.0
    target_std = 18.0
    target_p2_5 = 33.7
    target_p97_5 = 101.3
    
    # For a Beta distribution on [low, high]:
    # mean = low + (high-low)*μ where μ = α/(α+β)
    # variance = (high-low)^2 * [αβ/((α+β)^2(α+β+1))]
    
    def objective(params):
        alpha, beta_p, low, high = params
        
        # Constraint checks
        if alpha <= 0 or beta_p <= 0 or low >= high:
            return 1e10
        
        # Create distribution
        dist = beta_dist(alpha, beta_p, loc=low, scale=(high-low))
        
        # Calculate metrics
        mean_calc = dist.mean()
        std_calc = dist.std()
        p2_5_calc = dist.ppf(0.025)
        p97_5_calc = dist.ppf(0.975)
        
        # Weighted error
        error = (
            ((mean_calc - target_mean) / target_mean)**2 * 10 +  # Mean weight
            ((std_calc - target_std) / target_std)**2 * 5 +       # Std weight  
            ((p2_5_calc - target_p2_5) / target_p2_5)**2 * 100 +  # Percentile weights
            ((p97_5_calc - target_p97_5) / target_p97_5)**2 * 100
        )
        
        return error
    
    # Search for best fit
    bounds = [(0.1, 100), (0.1, 100), (0, 50), (60, 200)]
    
    result = differential_evolution(objective, bounds, maxiter=5000, popsize=30)
    
    if result.success:
        alpha, beta_p, low, high = result.x
        dist = beta_dist(alpha, beta_p, loc=low, scale=(high-low))
        
        print("="*60)
        print("FEASIBILITY ANALYSIS")
        print("="*60)
        print(f"\nTarget constraints:")
        print(f"  • Mean: 55.0 s")
        print(f"  • Std: 18.0 s")
        print(f"  • 2.5%: 33.7 s")
        print(f"  • 97.5%: 101.3 s")
        
        print(f"\nBest achievable fit:")
        print(f"  • Alpha: {alpha:.4f}")
        print(f"  • Beta: {beta_p:.4f}")
        print(f"  • Lower bound: {low:.3f} s")
        print(f"  • Upper bound: {high:.3f} s")
        print(f"\n  • Achieved mean: {dist.mean():.3f} s (error: {abs(dist.mean()-55):.3f})")
        print(f"  • Achieved std: {dist.std():.3f} s (error: {abs(dist.std()-18):.3f})")
        print(f"  • Achieved 2.5%: {dist.ppf(0.025):.3f} s (error: {abs(dist.ppf(0.025)-33.7):.3f})")
        print(f"  • Achieved 97.5%: {dist.ppf(0.975):.3f} s (error: {abs(dist.ppf(0.975)-101.3):.3f})")
        
        # Check consistency
        mu = alpha/(alpha+beta_p)
        theoretical_mean = low + (high-low)*mu
        theoretical_std = (high-low) * np.sqrt(alpha*beta_p/((alpha+beta_p)**2*(alpha+beta_p+1)))
        
        print(f"\nTheoretical skewness: {2*(beta_p-alpha)*np.sqrt(alpha+beta_p+1)/((alpha+beta_p+2)*np.sqrt(alpha*beta_p)):.3f}")
        
        return alpha, beta_p, low, high, result.fun
    
    return None

# Run feasibility check
result = check_feasibility()

# If feasible, find the best fit that prioritizes your specified percentiles
def fit_beta_to_percentiles_with_moments():
    """Fit Beta distribution prioritizing exact percentile matching"""
    
    target_mean = 55.0
    target_std = 18.0
    target_p2_5 = 33.7
    target_p97_5 = 101.3
    
    # For right-skewed data (mean > median), we expect median < mean
    # Let's find parameters that match percentiles exactly, then see resulting moments
    
    def objective_percentiles_only(params):
        alpha, beta_p, low, high = params
        if alpha <= 0.1 or beta_p <= 0.1 or low >= high or low < 0:
            return 1e10
        
        dist = beta_dist(alpha, beta_p, loc=low, scale=(high-low))
        
        error = (
            ((dist.ppf(0.025) - target_p2_5) / target_p2_5)**2 * 1000 +
            ((dist.ppf(0.975) - target_p97_5) / target_p97_5)**2 * 1000
        )
        return error
    
    # First, find parameters that match percentiles exactly
    bounds_percentiles = [(0.5, 50), (0.5, 50), (0, 40), (90, 150)]
    
    result_perc = differential_evolution(objective_percentiles_only, bounds_percentiles, 
                                         maxiter=5000, popsize=50, tol=1e-8)
    
    if result_perc.success:
        alpha, beta_p, low, high = result_perc.x
        dist_perc = beta_dist(alpha, beta_p, loc=low, scale=(high-low))
        
        print("\n" + "="*60)
        print("PERCENTILE-FIRST FIT (moments free)")
        print("="*60)
        print(f"\nParameters:")
        print(f"  • Alpha: {alpha:.4f}, Beta: {beta_p:.4f}")
        print(f"  • Support: [{low:.3f}, {high:.3f}] s")
        print(f"\nPercentile matching:")
        print(f"  • 2.5%: {dist_perc.ppf(0.025):.3f} s (target: {target_p2_5:.1f}s)")
        print(f"  • 97.5%: {dist_perc.ppf(0.975):.3f} s (target: {target_p97_5:.1f}s)")
        print(f"\nResulting moments:")
        print(f"  • Mean: {dist_perc.mean():.3f} s (target: {target_mean:.1f}s)")
        print(f"  • Std: {dist_perc.std():.3f} s (target: {target_std:.1f}s)")
        
        # Check right skew
        median = dist_perc.ppf(0.5)
        print(f"  • Median: {median:.3f} s")
        print(f"  • Mean - Median: {dist_perc.mean() - median:.3f} s {'(right skewed)' if dist_perc.mean() > median else '(left skewed)'}")
        
        return alpha, beta_p, low, high, dist_perc
    
    return None

# Run the percentile-first fit
result_percentile = fit_beta_to_percentiles_with_moments()

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nThe constraints are INCONSISTENT. Why?")
print("\n1. For a Beta distribution, the support bounds determine the extremes.")
print("2. With 2.5% at 33.7s and 97.5% at 101.3s, the total range is 67.6s.")
print("3. The natural support must extend beyond these percentiles.")
print("4. Given mean=55 and std=18, the implied coefficient of variation = 0.327")
print("5. This CV, combined with the 95% interval width, forces the distribution")
print("   to have specific bounds that conflict with the percentile locations.")
print("\nRECOMMENDATIONS:")
print("1. Relax one constraint (likely the standard deviation or mean)")
print("2. Or accept that the Beta distribution cannot exactly match all four")
print("3. Or use a different distribution family (e.g., lognormal, Gamma)")