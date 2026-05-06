"""
Physical and empirical constants for the QBiT sizing model.
All values from Kaneko & Martins (2023), Sec. III.
"""
import numpy as np

# Atmosphere
RHO_AIR = 1.225          # kg/m³  – sea-level air density

# Mission
T_HOVER = 60.0           # s      – hover time per takeoff/landing event

# QBiT airframe
BETA_QBIT  = 0.18        # –      – frame weight fraction
ETA_HOVER  = 0.65        # –      – hover figure of merit
CD0_WING   = 0.01        # –      – zero-lift wing drag coefficient
E_OSWALD   = 0.8         # –      – Oswald efficiency factor
AR_FIXED   = 8.0         # –      – fixed wing aspect ratio
N_ROTOR    = 4           # –      – number of rotors (quad)
SIGMA      = 0.13        # –      – rotor solidity
CD0_ROTOR  = 0.012       # –      – rotor blade zero-lift drag coefficient
KAPPA_MAX  = 1.15        # –      – induced power factor cap
BETA_CRUISE = np.radians(85.0)   # rad – fixed shaft tilt in cruise (5° AoA)

# Battery
BATTERY_DENSITY = 158.0  # Wh/kg  – pack energy density (paper Sec. III.A)
BATTERY_EFF     = 0.85   # –      – battery + transmission efficiency

# Weight regressions (Eqs. 4–7): coefficients give kg; multiply by g for N
G        = 9.81          # m/s²
K_MOTOR  = 2.506e-4      # kg/W   – motor weight regression
K_ESC    = 3.594e-4      # kg/W   – ESC weight regression
K_ROTOR_A = 0.7484       # kg/m²  – rotor weight regression (per rotor)
K_ROTOR_B = 0.0403       # kg/m   – rotor weight regression (per rotor)
K_WING_A  = -0.0802      # kg     – wing weight regression constant
K_WING_B  =  2.2854      # kg/m²  – wing weight regression slope

# Constraint limits (Table 1)
DL_MAX = 250.0           # N/m²   – max disk loading
BL_MAX = 0.14            # –      – max blade loading CT/σ
CL_MAX = 0.6             # –      – max cruise lift coefficient

# Design variable bounds (Table 1)
W_TOTAL_BOUNDS = (0.5 * G, 50.0 * G)   # N
V_INF_BOUNDS   = (10.0, 50.0)          # m/s
R_BOUNDS       = (0.05, 1.0)           # m
J_BOUNDS       = (0.01, 1.3)           # –
S_W_BOUNDS     = (0.05, 5.0)           # m²
