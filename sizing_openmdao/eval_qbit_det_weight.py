"""
Off-design feasibility check using actual OpenMDAO QBiT model.
Fixed design (geometry + MTOM), varying t_hover.
Uses build_qbit_model directly — no physics reimplementation.
"""
import numpy as np
import openmdao.api as om
import qbit.components.sizing_comps as sc
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import G, AR_FIXED, BATTERY_EFF, BATTERY_DENSITY, \
                           CL_MAX, DL_MAX, BL_MAX
import matplotlib.pyplot as plt
import warnings

# ── Mission parameters ────────────────────────────────────────────────────────
PAYLOAD_KG = 3.0
RANGE_M    = 15_000.0
N_C        = 2

# ── Three designs ─────────────────────────────────────────────────────────────
# Design MTOM = the MTOM the vehicle was sized to (at its design t_hover)
designs = {
    'Deterministic nominal\n(sized at t=55s)': {
        'x':         [31.36, 0.2227, 1.300, 0.1895],
        'mtom_kg':   6.981,
        'design_t':  55.0,
        'color':     'C0',
        'ls':        '--',
    },
    'UIDD worst-case\n(sized at t=101.5s)': {
        'x':         [29.33041342078393, 0.26979795053605055, 1.3, 0.24352624788169958],
        'mtom_kg':   7.858,
        'design_t':  101.5,
        'color':     'C1',
        'ls':        '-.',
    },
    'RBDO/UMDO\n(sized for mean + reliability)': {
        'x':         [31.78520562, 0.23156455, 1.3, 0.21064313],
        'mtom_kg':   7.103,
        'design_t':  55.0,   # mean hover time
        'color':     'C2',
        'ls':        '-',
    },
}

# ── Hover times to evaluate ───────────────────────────────────────────────────
T_EVAL = [25, 35, 45, 55, 65, 75, 85, 95, 101.5, 110, 120]

# ── Core evaluator ────────────────────────────────────────────────────────────

def run_model_at_fixed_W(V_inf, r, J, S_w, W_fixed_N,
                          t_hover, payload_kg, range_m, n_c):
    """
    Run QBiT OpenMDAO model with FIXED W_total and FIXED t_hover.
    Returns dict of outputs, or None on failure.
    
    This does NOT re-solve weight closure — W_total is injected directly.
    We read out what the model computes: E_req, W_battery_needed, CL, etc.
    """
    prob = om.Problem(reports=None)
    prob.model = build_qbit_model(payload_kg, range_m, n_c)
    prob.model.set_input_defaults('V_inf',   val=V_inf,   units='m/s')
    prob.model.set_input_defaults('r',       val=r,       units='m')
    prob.model.set_input_defaults('J',       val=J)
    prob.model.set_input_defaults('S_w',     val=S_w,     units='m**2')
    prob.model.set_input_defaults('W_total', val=W_fixed_N, units='N')
    prob.setup()

    # Inject t_hover into the module-level constant
    _orig = getattr(sc, 'T_HOVER', None)
    sc.T_HOVER = float(t_hover)

    try:
        prob.set_val('W_total', W_fixed_N)
        prob.set_val('V_inf',   V_inf)
        prob.set_val('r',       r)
        prob.set_val('J',       J)
        prob.set_val('S_w',     S_w)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.run_model()

        return {
            'E_req_Wh':      float(prob.get_val('E_req')[0]) / 3600.0,
            'W_bat_needed_kg': float(prob.get_val('W_battery')[0]) / G,
            'W_empty_kg':    float(prob.get_val('W_empty')[0]) / G,
            'P_hover_W':     float(prob.get_val('P_hover')[0]),
            'P_cruise_W':    float(prob.get_val('P_cruise')[0]),
            'cruise_CL':     float(prob.get_val('cruise_CL')[0]),
            'disk_loading':  float(prob.get_val('disk_loading')[0]),
            'blade_loading': float(prob.get_val('blade_loading')[0]),
            'weight_residual': float(prob.get_val('weight_residual')[0]),
        }
    except Exception as e:
        print(f"    Model failed at t={t_hover}: {e}")
        return None
    finally:
        if _orig is None:
            if hasattr(sc, 'T_HOVER'):
                del sc.T_HOVER
        else:
            sc.T_HOVER = _orig


# ── Main evaluation loop ──────────────────────────────────────────────────────

print("\n" + "="*95)
print("OFF-DESIGN FEASIBILITY CHECK")
print("Fixed design variables + fixed MTOM, varying operating t_hover")
print("Key question: does the battery have enough energy for each hover time?")
print("="*95)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
plot_data = {}

for name, cfg in designs.items():
    V_inf, r, J, S_w = cfg['x']
    W_fixed = cfg['mtom_kg'] * G

    # ── Get battery available from design point run ───────────────────
    # Run at design t_hover to get W_empty at the design point
    design_run = run_model_at_fixed_W(
        V_inf, r, J, S_w, W_fixed,
        cfg['design_t'], PAYLOAD_KG, RANGE_M, N_C
    )
    if design_run is None:
        print(f"\nFailed to evaluate design point for {name}")
        continue

    # Battery available = MTOM - W_empty - W_payload
    # (fixed by the design — this is what's physically in the vehicle)
    W_bat_available_kg = (cfg['mtom_kg'] 
                          - design_run['W_empty_kg'] 
                          - PAYLOAD_KG)
    E_available_Wh = W_bat_available_kg * BATTERY_DENSITY * BATTERY_EFF

    # ── Print header ──────────────────────────────────────────────────
    short_name = name.replace('\n', ' ')
    print(f"\n{'─'*95}")
    print(f"Design: {short_name}")
    print(f"  MTOM (fixed):        {cfg['mtom_kg']:.3f} kg")
    print(f"  W_empty (at design): {design_run['W_empty_kg']:.3f} kg")
    print(f"  W_battery available: {W_bat_available_kg:.3f} kg")
    print(f"  E_available:         {E_available_Wh:.1f} Wh")
    print(f"  Cruise CL (fixed):   {design_run['cruise_CL']:.4f} "
          f"({'✓' if design_run['cruise_CL'] <= CL_MAX else '✗ VIOLATED'})")
    print()
    print(f"  {'t_hover(s)':>10} {'E_req(Wh)':>12} {'W_bat_need(kg)':>16} "
          f"{'W_bat_avail(kg)':>17} {'Energy':>10} {'CL':>8} {'Margin(Wh)':>12}")
    print(f"  {'─'*90}")

    t_vals, e_req_vals, energy_ok_vals = [], [], []

    for t in T_EVAL:
        res = run_model_at_fixed_W(
            V_inf, r, J, S_w, W_fixed,
            t, PAYLOAD_KG, RANGE_M, N_C
        )
        if res is None:
            continue

        energy_ok  = res['W_bat_needed_kg'] <= W_bat_available_kg
        energy_str = '✓' if energy_ok else '✗ FAIL'
        cl_str     = '✓' if res['cruise_CL'] <= CL_MAX else '✗ FAIL'
        margin_Wh  = E_available_Wh - res['E_req_Wh']
        marker     = ' ← design pt' if abs(t - cfg['design_t']) < 3 else ''

        print(f"  {t:>10.1f} {res['E_req_Wh']:>12.1f} "
              f"{res['W_bat_needed_kg']:>16.3f} "
              f"{W_bat_available_kg:>17.3f} "
              f"{energy_str:>10} {cl_str:>8} "
              f"{margin_Wh:>12.1f}{marker}")

        t_vals.append(t)
        e_req_vals.append(res['E_req_Wh'])
        energy_ok_vals.append(energy_ok)

    plot_data[name] = {
        't':           np.array(t_vals),
        'E_req':       np.array(e_req_vals),
        'E_avail':     E_available_Wh,
        'color':       cfg['color'],
        'ls':          cfg['ls'],
    }

# ── Plots ─────────────────────────────────────────────────────────────────────
ax = axes[0]
t_fine = np.array(T_EVAL, dtype=float)

for name, pd in plot_data.items():
    label = name.replace('\n', ' ')
    ax.plot(pd['t'], pd['E_req'],
            color=pd['color'], ls=pd['ls'],
            linewidth=2, marker='o', markersize=4,
            label=f"{label} (E_req)")
    ax.axhline(pd['E_avail'],
               color=pd['color'], ls=':', linewidth=1.5, alpha=0.7,
               label=f"{label} (E_avail={pd['E_avail']:.0f}Wh)")

# Distribution reference lines
ax.axvline(55,    color='grey', ls='--', alpha=0.7, linewidth=1.5)
ax.axvline(101.5, color='grey', ls='-.', alpha=0.7, linewidth=1.5)
ax.text(56,   ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 100,
        'mean\n(55s)', fontsize=7, color='grey')
ax.text(102.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 100,
        '97.5%\n(101s)', fontsize=7, color='grey')

ax.set_xlabel('Operating hover time t_hover (s)')
ax.set_ylabel('Energy (Wh)')
ax.set_title('Energy Required vs Available\n'
             '(line = E_req, dotted = E_avail; cross = infeasible)')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

# Panel 2: Energy margin
ax = axes[1]
for name, pd in plot_data.items():
    label = name.replace('\n', ' ')
    margin = pd['E_avail'] - pd['E_req']
    ax.plot(pd['t'], margin,
            color=pd['color'], ls=pd['ls'],
            linewidth=2, marker='o', markersize=4,
            label=label)

ax.axhline(0, color='red', ls='-', linewidth=2, label='Feasibility limit')
ax.axvline(55,    color='grey', ls='--', alpha=0.7, linewidth=1.5)
ax.axvline(101.5, color='grey', ls='-.', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Operating hover time t_hover (s)')
ax.set_ylabel('Energy margin (Wh)  [positive = feasible]')
ax.set_title('Energy Margin vs Hover Time\n(above zero = design can complete mission)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle('Off-Design Feasibility: Fixed Design, Varying Hover Time',
             fontsize=11)
plt.tight_layout()
plt.savefig('sizing_openmdao/offdesign_feasibility.png',
            dpi=150, bbox_inches='tight')
plt.savefig('sizing_openmdao/offdesign_feasibility.svg',
            bbox_inches='tight')
print(f"\n{'='*95}")
print("Figure saved to sizing_openmdao/offdesign_feasibility.png")
plt.close()