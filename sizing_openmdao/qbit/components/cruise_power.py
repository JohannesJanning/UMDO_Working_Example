"""
CruisePowerComp - Eq. (11) of Kaneko & Martins (2023).

P_cruise = N · (T·V·sinβ + κ·T·Vᵢ + P₀)

κ (Eq. 12):  κ = min{ 1.15 ;  1/η - [√(2ρA) / T^1.5] · P₀ }
  NOTE: P₀ in the κ formula is the PROFILE power (Eq. 15),
        not the hover ideal power. P₀ must be computed in
        AerodynamicTrimComp before this component runs.

Ω = π·V∞/(r·J)  so r is needed here only for the κ formula (A = π·r²).

Partials: complex-step because of the conditional min() in κ.
"""
import numpy as np
import openmdao.api as om
from qbit.constants import RHO_AIR, ETA_HOVER, N_ROTOR, KAPPA_MAX, BETA_CRUISE


class CruisePowerComp(om.ExplicitComponent):
    """
    Inputs:  T_cruise [N/rotor], Vi [m/s], P0 [W/rotor], V_inf [m/s], r [m]
    Output:  P_cruise [W]   (total across all rotors)
    """

    def setup(self):
        self.add_input('T_cruise', val=2.0,   units='N',   desc='Cruise thrust per rotor')
        self.add_input('Vi',       val=0.20,  units='m/s', desc='Induced velocity')
        self.add_input('P0',       val=15.0,  units='W',   desc='Profile power per rotor')
        self.add_input('V_inf',    val=30.0,  units='m/s', desc='Cruise airspeed')
        self.add_input('r',        val=0.22,  units='m',   desc='Rotor radius')
        self.add_output('P_cruise', val=300.0, units='W',  desc='Total cruise shaft power')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        T  = inputs['T_cruise'][0]
        Vi = inputs['Vi'][0]
        P0 = inputs['P0'][0]
        V  = inputs['V_inf'][0]
        r  = inputs['r'][0]

        A = np.pi * r ** 2

        # κ – Eq. (12): uses profile power P0 (not hover ideal power)
        kappa_formula = 1.0 / ETA_HOVER - np.sqrt(2.0 * RHO_AIR * A) / T ** 1.5 * P0
        # complex-step-safe conditional: branch on real part only
        kappa = (KAPPA_MAX
                 if float(np.real(kappa_formula)) > KAPPA_MAX
                 else kappa_formula)

        P_per_rotor = (T * V * np.sin(BETA_CRUISE)
                       + kappa * T * Vi
                       + P0)
        outputs['P_cruise'] = N_ROTOR * P_per_rotor
