"""
Weight and power sizing components for the QBiT model.

InstalledPowerComp  - P_inst = 1.5 x P_hover  (hover dominates for QBiT)
EmptyWeightComp     - Eqs. (3)-(8): structural + propulsion empty weight
MissionEnergyComp   - Eq. (9): total mission energy (no climb segment)
BatteryWeightComp   - W_battery from E_req
"""
import numpy as np
import openmdao.api as om
from qbit.constants import (
    G, N_ROTOR, BETA_QBIT, T_HOVER,
    K_MOTOR, K_ESC, K_ROTOR_A, K_ROTOR_B, K_WING_A, K_WING_B,
    BATTERY_DENSITY, BATTERY_EFF,
)


# ── Installed power ───────────────────────────────────────────────────────

class InstalledPowerComp(om.ExplicitComponent):
    """
    P_inst = 1.5 x P_hover
    For QBiT, hover power always exceeds cruise power (wings carry the weight
    in cruise), so the 50% margin is applied to hover only (Sec. III.A).

    Inputs:  P_hover [W], P_cruise [W]
    Output:  P_inst  [W]
    """

    def setup(self):
        self.add_input('P_hover',  val=600.0, units='W')
        self.add_input('P_cruise', val=300.0, units='W')
        self.add_output('P_inst',  val=900.0, units='W',
                        desc='Installed motor power (50% margin on hover)')

    def setup_partials(self):
        self.declare_partials('P_inst', 'P_hover',  val=1.5)
        self.declare_partials('P_inst', 'P_cruise', val=0.0)

    def compute(self, inputs, outputs):
        outputs['P_inst'] = 1.5 * inputs['P_hover']


# ── Empty weight ──────────────────────────────────────────────────────────

class EmptyWeightComp(om.ExplicitComponent):
    """
    W_empty = W_motor + W_ESC + W_rotor + W_wing + W_frame

    Regression coefficients (Eqs. 4–7) give kg; multiply by g for Newtons.
    Rotor regression (Eq. 6) is per-rotor → multiply by N_ROTOR.
    Frame constant 0.5 in Eq. (8) is in kg → multiply by g.

    Inputs:  P_inst [W], r [m], S_w [m²], W_total [N]
    Output:  W_empty [N]
    """

    def setup(self):
        self.add_input('P_inst',  val=900.0, units='W')
        self.add_input('r',       val=0.22,  units='m')
        self.add_input('S_w',     val=0.20,  units='m**2')
        self.add_input('W_total', val=60.0,  units='N')
        self.add_output('W_empty', val=25.0, units='N', desc='Empty weight')

    def setup_partials(self):
        self.declare_partials('W_empty', ['P_inst', 'r', 'S_w', 'W_total'])

    def compute(self, inputs, outputs):
        P  = inputs['P_inst'][0]
        r  = inputs['r'][0]
        Sw = inputs['S_w'][0]
        W  = inputs['W_total'][0]

        W_motor = (K_MOTOR + K_ESC) * P * G                         # Eq. 4+5
        W_rotor = N_ROTOR * (K_ROTOR_A * r**2 - K_ROTOR_B * r) * G # Eq. 6 × N
        W_wing  = (K_WING_A + K_WING_B * Sw) * G                    # Eq. 7
        W_frame = 0.5 * G + BETA_QBIT * W                           # Eq. 8

        outputs['W_empty'] = W_motor + W_rotor + W_wing + W_frame

    def compute_partials(self, inputs, partials):
        r  = inputs['r'][0]
        P  = inputs['P_inst'][0]

        partials['W_empty', 'P_inst']  = (K_MOTOR + K_ESC) * G
        partials['W_empty', 'r']       = N_ROTOR * (2.0 * K_ROTOR_A * r - K_ROTOR_B) * G
        partials['W_empty', 'S_w']     = K_WING_B * G
        partials['W_empty', 'W_total'] = BETA_QBIT


# ── Mission energy ────────────────────────────────────────────────────────

class MissionEnergyComp(om.ExplicitComponent):
    """
    Eq. (9) without climb segment (Stage 0 simplification):
      E_req = P_hover · (2·(n_c+1)·t_hover) + P_cruise · (R / V∞)

    n_c is a fixed integer mission parameter, not a design variable.
    It is passed as a float input but rounded to int internally.

    Inputs:  P_hover [W], P_cruise [W], V_inf [m/s], R [m], n_c [–]
    Output:  E_req [J]
    """

    def setup(self):
        self.add_input('P_hover',  val=600.0, units='W')
        self.add_input('P_cruise', val=300.0, units='W')
        self.add_input('V_inf',    val=30.0,  units='m/s')
        self.add_input('R',        val=30e3,  units='m',  desc='Total route range')
        self.add_input('n_c',      val=1.0,               desc='Number of customers')
        self.add_output('E_req',   val=5e5,   units='J',  desc='Required mission energy')

    def setup_partials(self):
        self.declare_partials('E_req', ['P_hover', 'P_cruise', 'V_inf', 'R', 'n_c'])

    def compute(self, inputs, outputs):
        Ph = inputs['P_hover'][0]
        Pc = inputs['P_cruise'][0]
        V  = inputs['V_inf'][0]
        R  = inputs['R'][0]
        nc = int(round(float(inputs['n_c'][0])))   # fixed integer, safe scalar extract

        E_hover  = Ph * (2.0 * (nc + 1) * T_HOVER)
        E_cruise = Pc * (R / V)
        outputs['E_req'] = E_hover + E_cruise

    def compute_partials(self, inputs, partials):
        Pc = inputs['P_cruise'][0]
        V  = inputs['V_inf'][0]
        R  = inputs['R'][0]
        Ph = inputs['P_hover'][0]
        nc = int(round(float(inputs['n_c'][0])))

        partials['E_req', 'P_hover']  = 2.0 * (nc + 1) * T_HOVER
        partials['E_req', 'P_cruise'] = R / V
        partials['E_req', 'V_inf']    = -Pc * R / V**2
        partials['E_req', 'R']        = Pc / V
        partials['E_req', 'n_c']      = 2.0 * T_HOVER * Ph


# ── Battery weight ────────────────────────────────────────────────────────

class BatteryWeightComp(om.ExplicitComponent):
    """
    W_battery = (E_req / 3600) / (η_bat · ρ_b) · g

    Inputs:  E_req [J]
    Output:  W_battery [N]
    """

    def setup(self):
        self.add_input('E_req',     val=5e5,  units='J')
        self.add_output('W_battery', val=10.0, units='N', desc='Battery weight')

    def setup_partials(self):
        self.declare_partials('W_battery', 'E_req')

    def compute(self, inputs, outputs):
        E = inputs['E_req'][0]
        outputs['W_battery'] = (E / 3600.0) / (BATTERY_EFF * BATTERY_DENSITY) * G

    def compute_partials(self, inputs, partials):
        partials['W_battery', 'E_req'] = G / (3600.0 * BATTERY_EFF * BATTERY_DENSITY)
