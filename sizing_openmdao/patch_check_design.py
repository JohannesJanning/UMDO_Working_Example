

python3 - <<'EOF'
import numpy as np
import yaml, warnings, os, sys
sys.path.insert(0, '.')

from run_qbit_robust import inner_solve_for_Wtotal, sample_t_hover
from uqpce.pce.pce import PCE
from uqpce.pce.io import read_input_file
from uqpce.mdao import interface

# Recreate the UQPCE setup with Beta distribution
yaml_input = "/tmp/uqpce_diag.yaml"
matrix_file = "/tmp/uqpce_diag.dat"

import matplotlib; matplotlib.use('Agg')

config = {
    "Variable 0": {
        "name": "t_hover", "distribution": "beta",
        "alpha": 2.0576, "beta": 3.8263,
        "interval_low": 25.0, "interval_high": 110.0, "type": "aleatory",
    },
    "Settings": {"order": 2, "backend": "Agg", "track_convergence_off": True, "aleat_samp_size": 2000},
}
with open(yaml_input, 'w') as f:
    yaml.dump(config, f)

var_dict, settings = read_input_file(yaml_input)
settings.pop('plot', None); settings.pop('verbose', None)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    pce = PCE(outputs=False, plot=False, verbose=False, **settings)
    for v in var_dict.values(): pce.add_variable(**v)
    X = pce.sample()
    np.savetxt(matrix_file, X)

d = interface.initialize_dict(yaml_input, matrix_file)
t_quad = d['run_matrix'][:, 0]
print(f"Quadrature t_hover [s]: {t_quad.round(2)}")

# Evaluate W_total at these quadrature points using your robust design
x_robust = [29.79942897, 0.26030369, 1.3, 0.23397521]
W_vals = []
for t in t_quad:
    r = inner_solve_for_Wtotal(t, 3.0, 15000.0, 2, design_vars=tuple(x_robust))
    W_vals.append(r['W_total'] if isinstance(r, dict) else float('nan'))

W_arr = np.array(W_vals)
print(f"W_total at quad points [N]: {W_arr.round(3)}")
print(f"W_total range: [{W_arr.min():.2f}, {W_arr.max():.2f}] N")

# Now check resampled_var_basis range
rvb = d['resampled_var_basis']
print(f"\nresampled_var_basis shape: {rvb.shape}")
print(f"resampled_var_basis range: [{rvb.min():.4f}, {rvb.max():.4f}]")

# Compute PCE coefficients manually
vb = d['var_basis']
ns = d['norm_sq']
# coeffs = (var_basis.T @ var_basis)^-1 @ var_basis.T @ W / norm_sq  (simplified)
# Just show the resampled W distribution
coeffs = np.linalg.lstsq(vb, W_arr, rcond=None)[0]
W_resampled = rvb @ coeffs
print(f"\nResampled W_total [N]: min={W_resampled.min():.2f}, max={W_resampled.max():.2f}")
print(f"  mean={W_resampled.mean():.2f}, std={W_resampled.std():.2f}")
print(f"  95th pct={np.percentile(W_resampled, 95):.2f} N = {np.percentile(W_resampled, 95)/9.80665:.3f} kg")
print(f"\nFor sample_ref tuning:")
print(f"  sample_ref0 should be BELOW {W_resampled.min():.1f} N  → suggest {W_resampled.min()*0.95:.1f}")
print(f"  sample_ref  should be ABOVE {W_resampled.max():.1f} N  → suggest {W_resampled.max()*1.05:.1f}")
EOF