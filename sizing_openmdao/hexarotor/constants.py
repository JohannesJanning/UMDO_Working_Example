"""
Physical and empirical constants for the Hexarotor sizing model.
All values from Kaneko & Martins (2023), Sec. III.

Differences from QBiT (qbit/constants.py):
  N_ROTOR   = 6      (QBiT: 4)
  ETA_HOVER = 0.75   (QBiT: 0.65)
  BETA_HEX  = 0.20   (QBiT: 0.18) — frame weight fraction
  No wing term, no J design variable, no CL constraint.
  Design variable μ (edgewise advance ratio) replaces J.
"""
import numpy as np

# Atmosphere
RHO_AIR = 1.225          # kg/m³

# Mission timing
T_HOVER = 60.0           # s — hover time per takeoff/landing event

# Hexarotor airframe
BETA_HEX   = 0.20        # – frame weight fraction (higher than QBiT: no separate wing)
ETA_HOVER  = 0.75        # – hover figure of merit (higher than QBiT)
N_ROTOR    = 6           # – number of rotors
SIGMA      = 0.13        # – rotor solidity
CD0_ROTOR  = 0.012       # – blade zero-lift drag coefficient
KAPPA_MAX  = 1.15        # – induced power factor cap

# Battery
BATTERY_DENSITY = 158.0  # Wh/kg
BATTERY_EFF     = 0.85   # –

# Weight regression coefficients (Eqs. 4–6, 8) — same as QBiT, output in kg, ×g for N
G         = 9.81
K_MOTOR   = 2.506e-4     # kg/W
K_ESC     = 3.594e-4     # kg/W
K_ROTOR_A = 0.7484       # kg/m²  (per rotor)
K_ROTOR_B = 0.0403       # kg/m   (per rotor)
# No K_WING_* — hexarotor has no wing

# Constraint limits (Table 1)
DL_MAX = 250.0           # N/m²  disk loading
BL_MAX = 0.14            # –     blade loading CT/σ
# No CL_MAX — "CL ≤ 0.6, QBiT only" per Table 1

# Design variable bounds (Table 1)
W_TOTAL_BOUNDS = (0.5 * G, 50.0 * G)   # N
V_INF_BOUNDS   = (10.0, 50.0)          # m/s
R_BOUNDS       = (0.05, 1.0)           # m  rotor radius
MU_BOUNDS      = (0.01, 0.5)           # –  edgewise advance ratio (hexarotor only)
