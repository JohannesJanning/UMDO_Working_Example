"""
QBiT Conceptual Sizing Model
Based on: Kaneko & Martins (2023) "Fleet Design Optimization of Package Delivery
          Unmanned Aerial Vehicles Considering Operations", J. Aircraft, 60(4).

Physics and regression constants are taken directly from the paper (Sec. III).
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from dataclasses import dataclass
from typing import Optional
import warnings

# ---------------------------------------------------------------------------
# Physical & empirical constants (Kaneko & Martins 2023, Sec. III)
# ---------------------------------------------------------------------------
RHO_AIR    = 1.225        # kg/m³  – sea-level air density
T_HOVER    = 60.0         # s      – hover time per takeoff/landing operation
BETA_QBIT  = 0.18         # –      – frame weight fraction (QBiT)
ETA_HOVER  = 0.65         # –      – hover figure of merit (QBiT)
CD0_WING   = 0.01         # –      – zero-lift wing drag coefficient
CD0_BODY   = 0.1 + 0.2*np.cos(0.85)**3   # applied when β is known
E_OSWALD   = 0.8          # –      – Oswald efficiency
AR_FIXED   = 8.0          # –      – wing aspect ratio (fixed per paper)
N_ROTOR    = 4            # –      – number of rotors for QBiT (quad)
SIGMA      = 0.13         # –      – rotor solidity
CD0_ROTOR  = 0.012        # –      – airfoil zero-lift drag coefficient
KAPPA_MAX  = 1.15         # –      – max induced power factor cap
BATTERY_DENSITY = 158.0   # Wh/kg  – usable battery energy density
BATTERY_EFF     = 0.85    # –      – battery + transmission efficiency factor

# Motor/ESC regression (Eq. 4–5)
K_MOTOR = 2.506e-4        # kg/W
K_ESC   = 3.594e-4        # kg/W

# Rotor weight regression (Eq. 6)
K_ROTOR_A = 0.7484        # kg/m²
K_ROTOR_B = 0.0403        # kg/m

# Wing weight regression (Eq. 7)
K_WING_A = -0.0802        # kg
K_WING_B =  2.2854        # kg/m²

# Disk loading & blade-loading limits (Table 1)
DL_MAX    = 250.0         # N/m²
BL_MAX    = 0.14          # CT/sigma
CL_MAX    = 0.6           # cruise lift coefficient (QBiT only)

# Design variable bounds (Table 1)
BOUNDS = dict(
    W_total = (0.5 * 9.81, 50.0 * 9.81),   # N
    V_inf   = (10.0, 50.0),                 # m/s
    r       = (0.05, 1.0),                  # m
    J       = (0.01, 1.3),                  # propeller advance ratio
    S_w     = (0.05, 5.0),                  # m²
)


# ---------------------------------------------------------------------------
# Helper: solve rotor inflow equation (Eq. 13) for QBiT in cruise
# ---------------------------------------------------------------------------
def solve_inflow(mu: float, CT: float, beta: float) -> float:
    """Solve λ from: λ = μ·tan(β) + CT / (2·sqrt(μ²+λ²))"""
    def residual(lam):
        return lam - mu * np.tan(beta) - CT / (2.0 * np.sqrt(mu**2 + lam[0]**2))
    lam_init = CT / (2.0 * np.sqrt(mu**2 + (CT / 2.0)**2))
    sol = fsolve(residual, [lam_init], full_output=True)
    return float(sol[0][0])


# ---------------------------------------------------------------------------
# Weight residuals (Eqs. 2–8)
# ---------------------------------------------------------------------------
def compute_empty_weight(P_installed: float, r: float, S_w: float,
                         W_total: float) -> float:
    """Returns W_empty [N] given installed power P [W], rotor radius r [m],
    wing area S_w [m²], and total weight W_total [N]."""
    W_motor = K_MOTOR * P_installed * 9.81   # convert mass→weight
    W_ESC   = K_ESC   * P_installed * 9.81 
    W_rotor = (K_ROTOR_A * r**2 - K_ROTOR_B * r) * N_ROTOR * 9.81  # Eq. 6 (in kg→N)
    W_wing  = (K_WING_A  + K_WING_B * S_w) * 9.81          # Eq. 7
    W_frame = 0.5 * 9.81 + BETA_QBIT * W_total                            # Eq. 8
    return W_motor + W_ESC + W_rotor + W_wing + W_frame


# ---------------------------------------------------------------------------
# Power in hover (Eq. 10)
# ---------------------------------------------------------------------------
def hover_power(W_total: float, r: float) -> float:
    """Shaft power in hover [W], using momentum theory."""
    T_per_rotor = W_total / N_ROTOR
    A_disk = np.pi * r**2
    P_hover = (1.0 / ETA_HOVER) * (T_per_rotor**1.5) / np.sqrt(2.0 * RHO_AIR * A_disk)
    return N_ROTOR * P_hover


# ---------------------------------------------------------------------------
# Power in cruise for QBiT (Eqs. 11–15, trim from Eq. 18)
# ---------------------------------------------------------------------------
def cruise_power_qbit(W_total: float, r: float, V_inf: float, S_w: float,
                      J: float) -> tuple[float, float]:
    """Returns (P_cruise [W], beta_trim [rad]) for QBiT in winged cruise."""
    A_disk = np.pi * r**2
    n_rev  = V_inf / (2.0 * r * J)   # rotor rev from J definition
    Omega  = 2.0 * np.pi * n_rev

    # Trim: L = W, nrotor*T = D  →  iterate because D depends on CL & W
    # CL from lift equation: L = 0.5*rho*V²*S_w*CL = W_total
    CL = W_total / (0.5 * RHO_AIR * V_inf**2 * S_w)

    # Wing induced drag
    CDi = CL**2 / (np.pi * AR_FIXED * E_OSWALD)

    # Body drag (fixed β ≈ 85° = 5° AoA assumption from paper)
    beta_body = np.deg2rad(85.0)
    r_body    = 0.58 * r
    L_body    = 2.5 * 2.0 * r_body
    S_body    = r_body * L_body
    CD_b      = 0.1 + 0.2 * (np.cos(beta_body))**3
    D_body    = 0.5 * RHO_AIR * V_inf**2 * S_body * CD_b

    D_wing = 0.5 * RHO_AIR * V_inf**2 * S_w * (CD0_WING + CDi)
    D_total = D_body + D_wing

    # Thrust per rotor to overcome drag (nrotor * T = D)
    T_per_rotor = D_total / N_ROTOR

    CT = T_per_rotor / (RHO_AIR * A_disk * (Omega * r)**2)
    mu = V_inf * np.cos(beta_body) / (Omega * r)

    # Inflow ratio (Eq. 13)
    lam = solve_inflow(mu, CT, beta_body)

    # Induced velocity (Eq. 12)
    Vi = lam * Omega * r - V_inf * np.sin(beta_body)

    # Induced power factor (Eq. 12)
    # P0_ref = (1.0 / ETA_HOVER) * (T_per_rotor**1.5) / np.sqrt(2.0 * RHO_AIR * A_disk)
  
    # Profile power (Eq. 15)
    P0 = SIGMA * CD0_ROTOR / 8.0 * (1.0 + 4.65 * mu**2) * RHO_AIR * A_disk * (Omega * r)**3

    kappa  = min(KAPPA_MAX, 1.0 / ETA_HOVER - np.sqrt(2.0 * RHO_AIR * A_disk) / T_per_rotor**1.5 * P0)


    # Cruise shaft power per rotor (Eq. 11)
    P_rotor = T_per_rotor * V_inf * np.sin(beta_body) + kappa * T_per_rotor * Vi + P0
    P_cruise = N_ROTOR * P_rotor

    return P_cruise, float(beta_body)


# ---------------------------------------------------------------------------
# Energy budget (Eq. 9) – no climb segment (per Stage 0 note)
# ---------------------------------------------------------------------------
def mission_energy(P_hover: float, P_cruise: float,
                   V_inf: float, R: float, n_c: int) -> float:
    """Total mission energy [J]. Climb deleted per Stage 0."""
    E_hover  = P_hover * (2.0 * (n_c + 1) * T_HOVER)   # takeoff + landing at each stop
    E_cruise = P_cruise * (R / V_inf)
    return E_hover + E_cruise


# ---------------------------------------------------------------------------
# Battery weight
# ---------------------------------------------------------------------------
def battery_weight(E_req: float) -> float:
    """Battery weight [N] from required energy [J]."""
    E_wh = E_req / 3600.0
    m_bat = E_wh / (BATTERY_EFF * BATTERY_DENSITY)
    return m_bat * 9.81


# ---------------------------------------------------------------------------
# Sizing residual – the core equality constraint (Eq. 2)
# ---------------------------------------------------------------------------
def weight_residual(x: np.ndarray, W_payload: float,
                    R: float, n_c: int) -> np.ndarray:
    """
    Design vector x = [W_total, V_inf, r, J, S_w]
    Returns scalar residual for root-finding or equality constraint.
    """
    W_total, V_inf, r, J, S_w = x

    P_hover  = hover_power(W_total, r)
    P_cruise, _ = cruise_power_qbit(W_total, r, V_inf, S_w, J)

    # Installed power (50 % margin on max of hover / cruise)
    P_installed = 1.5 * P_hover #1.5 * max(P_hover, P_cruise)

    W_empty = compute_empty_weight(P_installed, r, S_w, W_total)

    E_req   = mission_energy(P_hover, P_cruise, V_inf, R, n_c)
    W_bat   = battery_weight(E_req)

    return W_total - (W_payload + W_bat + W_empty)


# ---------------------------------------------------------------------------
# Constraint functions for optimizer
# ---------------------------------------------------------------------------
def disk_loading(W_total: float, r: float) -> float:
    """T/A [N/m²] – must be ≤ DL_MAX."""
    T = W_total / N_ROTOR
    A = np.pi * r**2
    return T / A


def blade_loading(W_total: float, r: float, V_inf: float, J: float) -> float:
    """CT/σ – must be ≤ BL_MAX."""
    A = np.pi * r**2
    n_rev = V_inf / (2.0 * r * J)
    Omega = 2.0 * np.pi * n_rev
    T = W_total / N_ROTOR
    CT = T / (RHO_AIR * A * (Omega * r)**2)
    return CT / SIGMA


def cruise_CL(W_total: float, V_inf: float, S_w: float) -> float:
    """Lift coefficient in cruise – must be ≤ CL_MAX."""
    return W_total / (0.5 * RHO_AIR * V_inf**2 * S_w)


# ---------------------------------------------------------------------------
# QBiT Sizing Optimizer (Table 1 problem)
# ---------------------------------------------------------------------------
@dataclass
class MissionSpec:
    """Mission requirements fed to the sizing optimizer."""
    R: float          # mission range [m]
    W_payload: float  # total payload weight [N]
    n_c: int          # number of delivery customers

    def summary(self) -> str:
        return (f"R={self.R/1e3:.1f} km, "
                f"payload={self.W_payload/9.81:.2f} kg, "
                f"n_c={self.n_c}")


@dataclass
class SizingResult:
    """Output of the QBiT sizing optimization."""
    W_total:   float   # total takeoff weight [N]
    W_battery: float   # battery weight [N]
    W_empty:   float   # empty weight [N]
    P_hover:   float   # hover power [W]
    P_cruise:  float   # cruise power [W]
    V_inf:     float   # optimised cruise speed [m/s]
    r:         float   # rotor radius [m]
    J:         float   # propeller advance ratio
    S_w:       float   # wing area [m²]
    b:         float   # wingspan derived from S_w and AR [m]
    chord:     float   # mean chord derived from S_w and b [m]
    E_req:     float   # required mission energy [J]
    converged: bool    # optimiser converged flag

    def summary(self) -> str:
        lines = [
            f"  MTOM          : {self.W_total/9.81:7.3f} kg  ({self.W_total:.1f} N)",
            f"  Battery mass  : {self.W_battery/9.81:7.3f} kg",
            f"  Empty mass    : {self.W_empty/9.81:7.3f} kg",
            f"  Cruise speed  : {self.V_inf:7.2f} m/s",
            f"  Rotor radius  : {self.r:7.4f} m",
            f"  Wing area     : {self.S_w:7.4f} m²",
            f"  Wingspan      : {self.b:7.4f} m  (AR={AR_FIXED})",
            f"  Mean chord    : {self.chord:7.4f} m",
            f"  Prop adv. ratio J : {self.J:5.3f}",
            f"  P_hover       : {self.P_hover:8.1f} W",
            f"  P_cruise      : {self.P_cruise:8.1f} W",
            f"  E_required    : {self.E_req/3600:.3f} Wh",
            f"  Converged     : {self.converged}",
        ]
        return "\n".join(lines)


def size_qbit(mission: MissionSpec,
              x0: Optional[np.ndarray] = None,
              verbose: bool = False) -> SizingResult:
    """
    Minimise QBiT MTOM subject to:
      - weight residual equality (Eq. 2)
      - disk loading  T/A ≤ 250 N/m²
      - blade loading CT/σ ≤ 0.14
      - cruise CL     ≤ 0.6
      - box bounds from Table 1

    Design vector: x = [W_total, V_inf, r, J, S_w]
    """
    W_pl = mission.W_payload
    R    = mission.R
    n_c  = mission.n_c

    # Initial guess
    if x0 is None:
        x0 = np.array([
            5.0 * 9.81,   # W_total ~ 5 kg
            33.0,         # V_inf  ~ 33 m/s (QBiT default from paper)
            0.20,         # r ~ 20 cm
            1.3,          # J at upper bound (paper shows this is always active)
            0.3,          # S_w ~ 0.3 m²
        ])

    bounds = [
        BOUNDS['W_total'],
        BOUNDS['V_inf'],
        BOUNDS['r'],
        BOUNDS['J'],
        BOUNDS['S_w'],
    ]

    def objective(x):
        return x[0]   # minimise W_total

    constraints = [
        # Weight balance equality (Eq. 2) – treated as equality
        {
            'type': 'eq',
            'fun': lambda x: weight_residual(x, W_pl, R, n_c),
        },
        # Disk loading inequality T/A ≤ 250
        {
            'type': 'ineq',
            'fun': lambda x: DL_MAX - disk_loading(x[0], x[2]),
        },
        # Blade loading CT/σ ≤ 0.14
        {
            'type': 'ineq',
            'fun': lambda x: BL_MAX - blade_loading(x[0], x[2], x[1], x[3]),
        },
        # Cruise CL ≤ 0.6
        {
            'type': 'ineq',
            'fun': lambda x: CL_MAX - cruise_CL(x[0], x[1], x[4]),
        },
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 2000, 'disp': verbose},
        )

    W_total, V_inf, r, J, S_w = result.x

    P_hover  = hover_power(W_total, r)
    P_cruise, _ = cruise_power_qbit(W_total, r, V_inf, S_w, J)
    P_inst   = 1.5 * max(P_hover, P_cruise)
    W_empty  = compute_empty_weight(P_inst, r, S_w, W_total)
    E_req    = mission_energy(P_hover, P_cruise, V_inf, R, n_c)
    W_bat    = battery_weight(E_req)

    # Geometric derivations from AR = b²/S_w
    b     = np.sqrt(AR_FIXED * S_w)
    chord = S_w / b

    return SizingResult(
        W_total=W_total,
        W_battery=W_bat,
        W_empty=W_empty,
        P_hover=P_hover,
        P_cruise=P_cruise,
        V_inf=V_inf,
        r=r,
        J=J,
        S_w=S_w,
        b=b,
        chord=chord,
        E_req=E_req,
        converged=result.success,
    )


# ---------------------------------------------------------------------------
# Stage 1 – single UAV, single route (1 node), minimum MTOM
# ---------------------------------------------------------------------------
@dataclass
class Stage1Problem:
    """
    Simplest possible design-routing problem:
      - 1 UAV design k_1
      - 1 delivery node n_1 (depot → n_1 → depot)
      - minimise MTOM (proxy for cost + energy)
    """
    payload_kg: float     # package mass [kg]
    range_m:    float     # one-way distance depot → n_1 [m]

    def solve(self, verbose: bool = True) -> SizingResult:
        mission = MissionSpec(
            R        = 2.0 * self.range_m,   # round trip
            W_payload = self.payload_kg * 9.81,
            n_c       = 1,
        )
        if verbose:
            print("=" * 60)
            print("QBiT Conceptual Model – Stage 1")
            print("Single UAV · Single Route · Minimise MTOM")
            print("=" * 60)
            print(f"Mission: {mission.summary()}")
            print("-" * 60)

        result = size_qbit(mission, verbose=verbose)

        if verbose:
            print(result.summary())
            print("=" * 60)

        return result


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Stage 1 – deliver 3 kg to a node 15 km away (30 km round trip)
    problem = Stage1Problem(payload_kg=3.0, range_m=25_000.0)
    res = problem.solve(verbose=True)


# ---------------------------------------------------------------------------
# Stage 2 stub – two nodes, optional separate designs
# ---------------------------------------------------------------------------
@dataclass
class DeliveryNode:
    name:       str
    range_m:    float   # one-way distance from depot [m]
    payload_kg: float   # package mass for this node [kg]


@dataclass
class Stage2Problem:
    """
    Two-node problem.
    mode='shared'  → one design k_1 must serve both n_1 and n_2 (multimission)
    mode='separate' → allow two designs k_1, k_2 (one per node)
    """
    nodes: list  # list[DeliveryNode]
    mode:  str = 'separate'   # 'shared' | 'separate'

    def solve(self, verbose: bool = True) -> dict:
        if verbose:
            print("=" * 60)
            print(f"QBiT Conceptual Model – Stage 2 ({self.mode} design)")
            print("=" * 60)

        results = {}

        if self.mode == 'separate':
            for node in self.nodes:
                mission = MissionSpec(
                    R         = 2.0 * node.range_m,
                    W_payload  = node.payload_kg * 9.81,
                    n_c        = 1,
                )
                if verbose:
                    print(f"\nDesign for {node.name}: {mission.summary()}")
                    print("-" * 40)
                res = size_qbit(mission)
                results[node.name] = res
                if verbose:
                    print(res.summary())

        elif self.mode == 'shared':
            # Multimission: size one UAV to handle BOTH missions
            # Use the more demanding mission to size (conservative approach)
            # A proper implementation would loop the optimizer over both missions
            # simultaneously – left as an extension in Stage 2 development.
            missions = [
                MissionSpec(
                    R         = 2.0 * n.range_m,
                    W_payload  = n.payload_kg * 9.81,
                    n_c        = 1,
                )
                for n in self.nodes
            ]
            # Size for the combined worst-case: max range + total payload
            combined = MissionSpec(
                R         = max(m.R for m in missions),
                W_payload  = sum(m.W_payload for m in missions),
                n_c        = len(self.nodes),
            )
            if verbose:
                print(f"\nShared design mission (conservative sizing): {combined.summary()}")
                print("-" * 40)
            res = size_qbit(combined)
            results['shared_k1'] = res
            if verbose:
                print(res.summary())

        if verbose:
            fleet_mtom = sum(r.W_total / 9.81 for r in results.values())
            print(f"\nFleet total MTOM: {fleet_mtom:.3f} kg")
            print("=" * 60)

        return results