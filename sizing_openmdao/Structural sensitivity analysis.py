"""
Structural sensitivity analysis — CORRECT version.
Re-solves W_total at each t_hover for fixed geometry.
This captures the full W_total → W_empty feedback loop.
Shows: how much of MTOM growth with t_hover is structural vs battery.
"""
import numpy as np
import matplotlib.pyplot as plt
from run_qbit_MCS import inner_solve_for_Wtotal, G

PAYLOAD_KG = 3.0
RANGE_M    = 15_000.0
N_C        = 2

designs = {
    'Deterministic nominal': {
        'x':     [31.36, 0.2227, 1.300, 0.1895],
        'color': 'C0', 'ls': '--',
    },
    'UIDD worst-case': {
        'x':     [29.31, 0.2703, 1.300, 0.2441],
        'color': 'C1', 'ls': '-.',
    },
    'RBDO/UMDO': {
        'x':     [31.77, 0.2314, 1.300, 0.2102],
        'color': 'C2', 'ls': '-',
    },
}

T_SWEEP = np.linspace(25, 130, 40)

print("\n" + "="*75)
print("STRUCTURAL SENSITIVITY: Re-solved W_total at each t_hover")
print("Fixed geometry (V_inf, r, J, S_w), weight closure re-solved")
print("="*75)

all_data = {}

for name, cfg in designs.items():
    print(f"\n{name} ...")
    x = cfg['x']

    W_total_arr = []
    W_empty_arr = []
    W_bat_arr   = []
    t_valid     = []

    for t in T_SWEEP:
        res = inner_solve_for_Wtotal(
            t_hover_sample=t,
            payload_kg=PAYLOAD_KG,
            range_m=RANGE_M,
            n_c=N_C,
            design_vars=tuple(x)
        )
        if not isinstance(res, dict):
            continue

        W_total_arr.append(res['W_total'] / G)
        W_empty_arr.append(res['W_empty'] / G)
        W_bat_arr.append(res['W_battery'] / G)
        t_valid.append(t)

    t_arr       = np.array(t_valid)
    W_total_arr = np.array(W_total_arr)
    W_empty_arr = np.array(W_empty_arr)
    W_bat_arr   = np.array(W_bat_arr)

    # How much does each component grow over the full range?
    dW_total = W_total_arr[-1] - W_total_arr[0]
    dW_empty = W_empty_arr[-1] - W_empty_arr[0]
    dW_bat   = W_bat_arr[-1]   - W_bat_arr[0]

    print(f"  t range: {t_arr[0]:.0f}s → {t_arr[-1]:.0f}s")
    print(f"  ΔW_total : {dW_total:+.4f} kg  (100%)")
    print(f"  ΔW_empty : {dW_empty:+.4f} kg  "
          f"({dW_empty/dW_total*100:.1f}% of total growth)")
    print(f"  ΔW_bat   : {dW_bat:+.4f}  kg  "
          f"({dW_bat/dW_total*100:.1f}% of total growth)")

    all_data[name] = {
        't':       t_arr,
        'W_total': W_total_arr,
        'W_empty': W_empty_arr,
        'W_bat':   W_bat_arr,
        'color':   cfg['color'],
        'ls':      cfg['ls'],
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Panel 1: Stacked area — W_empty vs W_battery contribution to MTOM
ax = axes[0]
for name, d in all_data.items():
    ax.plot(d['t'], d['W_total'],
            color=d['color'], ls=d['ls'], lw=2.5,
            label=f"{name} (MTOM)")
    ax.plot(d['t'], d['W_empty'],
            color=d['color'], ls=d['ls'], lw=1.0,
            alpha=0.5, label=f"{name} (W_empty)")

ax.axvline(55,    color='grey', ls='--', alpha=0.7, lw=1.5, label='mean (55s)')
ax.axvline(101.5, color='grey', ls='-.', alpha=0.7, lw=1.5, label='97.5% (101.5s)')
ax.set_xlabel('t_hover (s)')
ax.set_ylabel('Mass (kg)')
ax.set_title('MTOM and W_empty vs hover time\n'
             '(thick = MTOM, thin = W_empty)')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Panel 2: W_battery only — the variable component
ax = axes[1]
for name, d in all_data.items():
    ax.plot(d['t'], d['W_bat'],
            color=d['color'], ls=d['ls'], lw=2.5,
            label=name)

ax.axvline(55,    color='grey', ls='--', alpha=0.7, lw=1.5, label='mean (55s)')
ax.axvline(101.5, color='grey', ls='-.', alpha=0.7, lw=1.5, label='97.5% (101.5s)')
ax.set_xlabel('t_hover (s)')
ax.set_ylabel('Battery mass (kg)')
ax.set_title('Battery mass vs hover time\n'
             '(dominant variable component)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle('Weight decomposition: structural vs battery sensitivity to t_hover\n'
             '(weight closure re-solved at each point)',
             fontsize=11)
plt.tight_layout()
plt.savefig('sizing_openmdao/structural_sensitivity_resolved.png',
            dpi=150, bbox_inches='tight')
plt.savefig('sizing_openmdao/structural_sensitivity_resolved.svg',
            bbox_inches='tight')
print("\nFigure saved.")
plt.close()