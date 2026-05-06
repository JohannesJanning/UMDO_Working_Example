"""
HoverPowerComp - Eq. (10), hexarotor variant.

P_hover_per_rotor = (1/η) · T^1.5 / sqrt(2·ρ·A)
P_hover_total     = N_rotor · P_hover_per_rotor
T = W_total / N_rotor   (hover trim)

η = 0.75 for hexarotor (vs 0.65 for QBiT).
N_rotor = 6.
"""
import numpy as np
import openmdao.api as om
from hexarotor.constants import RHO_AIR, ETA_HOVER, N_ROTOR, G


class HoverPowerComp(om.ExplicitComponent):
    """
    Inputs:  W_total [N], r [m]
    Output:  P_hover [W]
    """

    def setup(self):
        self.add_input('W_total', val=50.0,  units='N')
        self.add_input('r',       val=0.25,  units='m')
        self.add_output('P_hover', val=400.0, units='W')

    def setup_partials(self):
        self.declare_partials('P_hover', ['W_total', 'r'])

    def compute(self, inputs, outputs):
        W = inputs['W_total'][0]
        r = inputs['r'][0]
        T = W / N_ROTOR
        A = np.pi * r**2
        P_per = (1.0 / ETA_HOVER) * T**1.5 / np.sqrt(2.0 * RHO_AIR * A)
        outputs['P_hover'] = N_ROTOR * P_per

    def compute_partials(self, inputs, partials):
        W = inputs['W_total'][0]
        r = inputs['r'][0]
        T = W / N_ROTOR
        A = np.pi * r**2
        denom = np.sqrt(2.0 * RHO_AIR * A)
        dP_dT = N_ROTOR * (1.0 / ETA_HOVER) * 1.5 * T**0.5 / denom
        partials['P_hover', 'W_total'] = dP_dT / N_ROTOR
        dP_dA = N_ROTOR * (1.0 / ETA_HOVER) * T**1.5 * (-0.5) * (2.0 * RHO_AIR * A)**(-1.5) * 2.0 * RHO_AIR
        partials['P_hover', 'r'] = dP_dA * 2.0 * np.pi * r
