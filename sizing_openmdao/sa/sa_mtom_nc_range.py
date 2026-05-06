import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from matplotlib.lines import Line2D

# --- PATH FIX FOR SUBFOLDER ---
# This allows the script to find run_qbit and run_hexarotor in the parent folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

om.config_reports = False

from qbit.constants import G as G_QBiT
from hexarotor.constants import G as G_Hex  # both are 9.80665, but use explicit

# Import problem builders
from run_qbit import build_problem as build_qbit_problem
from run_hexarotor import build_problem as build_hex_problem   # hexarotor's build_problem


def run_parameter_sweep():
    # --- Configuration ---
    payload_kg = 3.0
    n_c_values = np.arange(1, 6)          # 1 to 5 customers
    ranges_km = [5, 10, 15, 20, 25, 30]   # mission ranges in km

    # Storage for results: two dictionaries (QBiT and Hexarotor)
    results_qbit = {r_km: [] for r_km in ranges_km}
    results_hex  = {r_km: [] for r_km in ranges_km}

    print(f"Starting sweep for Payload = {payload_kg} kg (both QBiT and Hexarotor)...")

    for r_km in ranges_km:
        range_m = r_km * 1000.0
        print(f"  Computing range: {r_km} km", end=" ", flush=True)

        for n_c in n_c_values:
            # -------- QBiT model --------
            try:
                prob_q = build_qbit_problem(payload_kg, range_m, n_c=n_c)
                prob_q.run_driver()
                converged_q = prob_q.driver.result.success
                if converged_q:
                    mtom_q_kg = prob_q.get_val('W_total')[0] / G_QBiT
                else:
                    mtom_q_kg = np.nan
            except Exception:
                mtom_q_kg = np.nan
            results_qbit[r_km].append(mtom_q_kg)

            # -------- Hexarotor model --------
            try:
                prob_h = build_hex_problem(payload_kg, range_m, n_c=n_c)
                prob_h.run_driver()
                converged_h = prob_h.driver.result.success
                if converged_h:
                    mtom_h_kg = prob_h.get_val('W_total')[0] / G_Hex
                else:
                    mtom_h_kg = np.nan
            except Exception:
                mtom_h_kg = np.nan
            results_hex[r_km].append(mtom_h_kg)

            print(".", end="", flush=True)
        print(" Done.")

    # --- Plotting (both models on one figure) ---
    plt.figure(figsize=(12, 8))

    colors = plt.get_cmap('tab10').colors

    # Plot each range individually, including fill_between
    for i, r_km in enumerate(ranges_km):
        # QBiT (solid, circles)
        plt.plot(n_c_values, results_qbit[r_km], 'o-',
                color=colors[i], linewidth=1.8, markersize=6,
                label=f'QBiT, range = {2*r_km} km')
        
        # Hexarotor (dashed, squares)
        plt.plot(n_c_values, results_hex[r_km], 's--',
                color=colors[i], linewidth=1.8, markersize=5,
                label=f'Hexarotor, range = {2*r_km} km')
        
        # Mask where QBiT < Hexarotor (fill between them)
        qbit_vals = np.array(results_qbit[r_km])
        hex_vals = np.array(results_hex[r_km])
        mask = qbit_vals < hex_vals
        
        plt.fill_between(n_c_values, qbit_vals, hex_vals,
                        where=mask,
                        color='gray', alpha=0.2,
                        interpolate=True)

    # Styling
    plt.title(f'MTOM Comparison - QBiT vs Hexarotor (Payload = {payload_kg} kg)',
            fontsize=14, fontweight='bold')
    plt.xlabel('Number of Nodes ($n_c$ - number of customers)', fontsize=12)
    plt.ylabel('MTOM (kg)', fontsize=12)

    plt.xticks(n_c_values)
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Custom legends (two separate) ---
    # Legend for line styles (configurations)
    legend_lines = [
        Line2D([0], [0], color='black', linestyle='-', marker='o', label='QBiT'),
        Line2D([0], [0], color='black', linestyle='--', marker='s', label='Hexarotor')
    ]
    legend1 = plt.legend(handles=legend_lines, title='Configuration',
                        loc='upper left', frameon=True)
    plt.gca().add_artist(legend1)

    # Legend for colors (ranges)
    legend_colors = [
        Line2D([0], [0], color=colors[i], lw=2, label=f'{2*r_km} km')
        for i, r_km in enumerate(ranges_km)
    ]
    plt.legend(handles=legend_colors, title='Mission Range',
            loc='upper right', frameon=True)

    plt.tight_layout()

    # This creates sa/results_sa relative to where this script sits
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, 'results_sa')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    save_path = os.path.join(output_dir, 'mtom_comparison_qbit_hex_range_nc.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_parameter_sweep()