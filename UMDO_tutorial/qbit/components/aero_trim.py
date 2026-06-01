"""
AerodynamicTrimComp - Eqs. (13)-(15), (17)-(18) of Kaneko & Martins (2023).

Computes the cruise trim state for the QBiT:
  - Lift trim: L = W  →  CL, CDi
  - Drag:  D_total = D_body + D_wing
  - Rotor thrust: T_cruise = D_total / N_rotor
  - Dimensionless rotor params: CT, μ  (Eq. 14)
  - Inflow ratio λ (Eq. 13) - solved via internal Newton loop
  - Induced velocity Vi (Eq. 12)
  - Profile power P0 (Eq. 15)

β = 85° is FIXED for the QBiT (paper Sec. III.B.3: "AoA of 5 deg assumed").
No outer iteration on β is required.

Partials: complex-step (method='cs') because of the internal Newton loop
for λ and the nonlinear drag expressions.
"""
import numpy as np
import openmdao.api as om
from qbit.constants import (RHO_AIR, N_ROTOR, AR_FIXED, E_OSWALD,
                             CD0_WING, SIGMA, CD0_ROTOR, BETA_CRUISE)


def _solve_lambda(mu, CT, beta, tol=1e-12, maxiter=200):
    """Newton solve for inflow ratio λ (Eq. 13), complex-step compatible.

    Equation:  λ = μ·tan(β) + CT / (2·√(μ²+λ²))
    Residual:  f(λ) = λ - μ·tan(β) - CT/(2·√(μ²+λ²)) = 0
    """
    lam = CT / (2.0 * np.sqrt(mu ** 2 + (CT / 2.0) ** 2))   # initial guess
    for _ in range(maxiter):
        g   = np.sqrt(mu ** 2 + lam ** 2)
        f   = lam - mu * np.tan(beta) - CT / (2.0 * g)
        df  = 1.0 + CT * lam / (2.0 * g ** 3)               # df/dλ
        step = f / df
        lam  = lam - step
        if np.abs(step) < tol:
            break
    return lam


class AerodynamicTrimComp(om.ExplicitComponent):
    """
    Inputs:  W_total [N], r [m], V_inf [m/s], S_w [m²], J [-]
    Outputs: T_cruise [N/rotor], CL, CT, mu, lam, Vi [m/s], P0 [W/rotor]
    """

    def setup(self):
        self.add_input('W_total', val=60.0,  units='N')
        self.add_input('r',       val=0.22,  units='m')
        self.add_input('V_inf',   val=30.0,  units='m/s')
        self.add_input('S_w',     val=0.20,  units='m**2')
        self.add_input('J',       val=1.3,   desc='Propeller advance ratio')

        self.add_output('T_cruise', val=2.0,  units='N',   desc='Cruise thrust per rotor')
        self.add_output('CL',       val=0.3,               desc='Cruise lift coefficient')
        self.add_output('CT',       val=0.002,             desc='Rotor thrust coefficient')
        self.add_output('mu',       val=0.04,              desc='Edgewise advance ratio')
        self.add_output('lam',      val=0.40,              desc='Inflow ratio')
        self.add_output('Vi',       val=0.20, units='m/s', desc='Induced velocity')
        self.add_output('P0',       val=15.0, units='W',   desc='Profile power per rotor')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        W   = inputs['W_total'][0]
        r   = inputs['r'][0]
        V   = inputs['V_inf'][0]
        S_w = inputs['S_w'][0]
        J   = inputs['J'][0]

        beta  = BETA_CRUISE
        A     = np.pi * r ** 2
        Omega = np.pi * V / (r * J)        # Ω = 2π·n_rev, n_rev = V/(2rJ)

        # ── Lift trim → aerodynamic drag ──────────────────────────────
        CL  = W / (0.5 * RHO_AIR * V ** 2 * S_w)
        CDi = CL ** 2 / (np.pi * AR_FIXED * E_OSWALD)

        # Body (cylinder) drag – Eq. (17)
        r_b  = 0.58 * r
        S_b  = r_b * (2.5 * 2.0 * r_b)     # Sb = r_body × L_body
        CD_b = 0.1 + 0.2 * np.cos(beta) ** 3
        D_b  = 0.5 * RHO_AIR * V ** 2 * S_b * CD_b

        # Wing drag – Eq. (18)
        D_w = 0.5 * RHO_AIR * V ** 2 * S_w * (CD0_WING + CDi)

        T_cr = (D_b + D_w) / N_ROTOR       # cruise thrust per rotor

        # ── Dimensionless rotor params (Eq. 14) ──────────────────────
        CT  = T_cr / (RHO_AIR * A * (Omega * r) ** 2)
        mu  = V * np.cos(beta) / (Omega * r)

        # ── Inflow λ (Eq. 13) ────────────────────────────────────────
        lam = _solve_lambda(mu, CT, beta)

        # ── Induced velocity (Eq. 12) ────────────────────────────────
        Vi  = lam * Omega * r - V * np.sin(beta)

        # ── Profile power (Eq. 15) ───────────────────────────────────
        P0  = (SIGMA * CD0_ROTOR / 8.0
               * (1.0 + 4.65 * mu ** 2)
               * RHO_AIR * A * (Omega * r) ** 3)

        outputs['T_cruise'] = T_cr
        outputs['CL']       = CL
        outputs['CT']       = CT
        outputs['mu']       = mu
        outputs['lam']      = lam
        outputs['Vi']       = Vi
        outputs['P0']       = P0
