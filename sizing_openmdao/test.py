import math
from scipy.stats import lognorm

# Recreate the distribution (same parameters as in your code)
target_mean, target_std, shift = 55.0, 18.0, 25.0
mu_prime = target_mean - shift
v_prime  = target_std ** 2
s_sq     = math.log(v_prime / mu_prime**2 + 1)
s        = math.sqrt(s_sq)
scale    = mu_prime / math.sqrt(v_prime / mu_prime**2 + 1)
dist     = lognorm(s, loc=shift, scale=scale)

print(f"T_HOVER mean (check): {dist.mean():.4f} s")   # should be ~55.0
print(f"T_HOVER 2.5th pct  : {dist.ppf(0.025):.6f} s")
print(f"T_HOVER 97.5th pct : {dist.ppf(0.975):.6f} s")  # this is your CI upper