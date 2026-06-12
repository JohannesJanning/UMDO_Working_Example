import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from matplotlib.lines import Line2D
from matplotlib import rcParams
import csv

# --- Journal Style Configuration (Enhanced Legibility) ---
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.2  # Slightly thicker frame
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# Large font sizes for publication
LABEL_SIZE = 16
TITLE_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 12

# --- PATH FIX ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
om.config_reports = False

from qbit.constants import G as G_QBiT
from hexarotor.constants import G as G_Hex
from run_qbit import build_problem as build_qbit_problem
from run_hexarotor import build_problem as build_hex_problem

def find_crossover(x_vals, y1_vals, y2_vals):
    """Finds the x-value where y1 becomes less than y2 using linear interpolation."""
    y1 = np.array(y1_vals)
    y2 = np.array(y2_vals)
    diff = y1 - y2
    
    for i in range(len(diff) - 1):
        # Check if a sign change occurred (y1 was > y2, now y1 < y2)
        if diff[i] > 0 and diff[i+1] < 0:
            # Linear interpolation formula for the zero crossing
            x_cross = x_vals[i] - diff[i] * (x_vals[i+1] - x_vals[i]) / (diff[i+1] - diff[i])
            return x_cross
    return None



def run_parameter_sweep():
    # --- Configuration ---
    payload_kg = 5.0
    n_c_values = np.arange(1, 6)             
    ranges_km = np.arange(5, 31, 5)        

    results_qbit = {nc: [] for nc in n_c_values}
    results_hex  = {nc: [] for nc in n_c_values}

    # (Simulation logic)
    print("Running optimizations...")
    for r_km in ranges_km:
        range_m = r_km * 1000.0
        for n_c in n_c_values:
            try:
                prob_q = build_qbit_problem(payload_kg, range_m, n_c=n_c)
                prob_q.run_driver()
                mtom_q_kg = prob_q.get_val('W_total')[0] / G_QBiT if prob_q.driver.result.success else np.nan
            except: mtom_q_kg = np.nan
            results_qbit[n_c].append(mtom_q_kg)
            try:
                prob_h = build_hex_problem(payload_kg, range_m, n_c=n_c)
                prob_h.run_driver()
                mtom_h_kg = prob_h.get_val('W_total')[0] / G_Hex if prob_h.driver.result.success else np.nan
            except: mtom_h_kg = np.nan
            results_hex[n_c].append(mtom_h_kg)

# --- Directory Setup ---
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, 'results_sa')
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV Data Processing ---
    csv_path = os.path.join(output_dir, 'breakeven_points_mtom_range_nc.csv')
    breakeven_data = []

    for n_c in n_c_values:
        q_vals = np.array(results_qbit[n_c])
        h_vals = np.array(results_hex[n_c])
        
        cross_r = find_crossover(ranges_km, q_vals, h_vals)
        
        if cross_r:
            # Interpolate the MTOM at the crossover point 
            # (Since it's the breakeven point, QBiT and Hex mass are equal here)
            mtom_at_cross = np.interp(cross_r, ranges_km, q_vals)
            
            breakeven_data.append({
                'n_c': n_c,
                'breakeven_mtom_kg': round(mtom_at_cross, 2),
                'total_range_km': round(cross_r * 2, 2)
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_c', 'breakeven_mtom_kg', 'total_range_km'])
        writer.writeheader()
        writer.writerows(breakeven_data)
    print(f"Breakeven CSV saved to: {csv_path}")




    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 8)) 
    
    colors = plt.get_cmap('tab10').colors

    for i, n_c in enumerate(n_c_values):
        q_vals = np.array(results_qbit[n_c])
        h_vals = np.array(results_hex[n_c])
        
        ax.plot(ranges_km, q_vals, linestyle='-', color=colors[i], 
                linewidth=2.0)
        
        ax.plot(ranges_km, h_vals, linestyle='--', color=colors[i], 
                linewidth=1.8, alpha=0.8)
        
        mask = q_vals < h_vals
        ax.fill_between(ranges_km, q_vals, h_vals, where=mask, 
                        color=colors[i], alpha=0.1, interpolate=True)

    # --- Scaling Font Sizes ---
    ax.set_title(f'Sizing Sensitivity (Payload Mass: $m_{{pay}} = {payload_kg}$ kg)', fontsize=TITLE_SIZE, pad=15)
    ax.set_xlabel('Total Mission Range, $R$ [km]', fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel('Maximum Take-off Mass (MTOM) [kg]', fontsize=LABEL_SIZE, labelpad=10)
    
    # Doubled Tick Labels with larger numbers
    ax.set_xticks(ranges_km)
    ax.set_xticklabels([f"{int(r*2)}" for r in ranges_km], fontsize=TICK_SIZE)
    
    # Update y-tick font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.2, size=6)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # --- Dual Legends (Larger Font) ---
    line_styles = [
        Line2D([0], [0], color='black', linestyle='-', label='QBiT (Transition)', lw=2),
        Line2D([0], [0], color='black', linestyle='--', label='Hexarotor (Multirotor)', lw=2)
    ]
    leg1 = ax.legend(handles=line_styles, loc='upper left', fontsize=LEGEND_SIZE, 
                     frameon=True, title='Vehicle Architecture', title_fontsize=LEGEND_SIZE+2)
    ax.add_artist(leg1)

    nc_legend = [
        Line2D([0], [0], color=colors[i], lw=3, label=f'$n_c = {n_c}$')
        for i, n_c in enumerate(n_c_values)
    ]
    ax.legend(handles=nc_legend, loc='lower right', fontsize=LEGEND_SIZE, ncol=2, 
              frameon=True, title='Customer Count', title_fontsize=LEGEND_SIZE+2)

    plt.tight_layout()

    # --- High-Resolution Output ---
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, 'results_sa')
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, 'figure_mtom_range_nc.png'), dpi=400, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_parameter_sweep()