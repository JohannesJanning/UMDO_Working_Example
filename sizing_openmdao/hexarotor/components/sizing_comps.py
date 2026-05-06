"""
Sizing components for the hexarotor model.

InstalledPowerComp   - P_inst = 1.5 * max(P_hover, P_cruise)
                       For hexarotor, cruise power can exceed hover,
                       so max() is used (unlike QBiT where hover dominates).
EmptyWeightComp      - Eqs. (3–6, 8): NO wing term.
MissionEnergyComp    - Eq. (9), identical to QBiT.
BatteryWeightComp    - identical to QBiT.
WeightResidualComp   - Eq. (2) equality constraint residual.
"""
import numpy as np
import openmdao.api as om
from hexarotor.constants import (
    G, N_ROTOR, BETA_HEX, T_HOVER,
    K_MOTOR, K_ESC, K_ROTOR_A, K_ROTOR_B,
    BATTERY_DENSITY, BATTERY_EFF,
)


# ── Installed power ───────────────────────────────────────────────────────

class InstalledPowerComp(om.ExplicitComponent):
    """
    P_inst = 1.5 * max(P_hover, P_cruise)

    For hexarotor, cruise can dominate at longer ranges because there is
    no wing to carry weight — rotors must provide both lift and thrust.
    max() ensures the motor is sized for whichever phase is more demanding.

    Inputs:  P_hover [W], P_cruise [W]
    Output:  P_inst  [W]
    """

    def setup(self):
        self.add_input('P_hover',  val=400.0, units='W')
        self.add_input('P_cruise', val=250.0, units='W')
        self.add_output('P_inst',  val=600.0, units='W')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Ph = inputs['P_hover'][0]
        Pc = inputs['P_cruise'][0]
        outputs['P_inst'] = 1.5 * (Ph if float(np.real(Ph)) >= float(np.real(Pc)) else Pc)


# ── Empty weight ──────────────────────────────────────────────────────────

class EmptyWeightComp(om.ExplicitComponent):
    """
    W_empty = W_motor + W_ESC + W_rotor + W_frame   (no wing term)

    Regression coefficients (Eqs. 4–6) give kg; multiply by g for Newtons.
    Rotor regression (Eq. 6) is per-rotor → multiply by N_ROTOR = 6.
    Frame constant 0.5 in Eq. (8) is in kg → multiply by g.
    β = 0.20 for hexarotor.

    Inputs:  P_inst [W], r [m], W_total [N]
    Output:  W_empty [N]
    """

    def setup(self):
        self.add_input('P_inst',  val=600.0, units='W')
        self.add_input('r',       val=0.25,  units='m')
        self.add_input('W_total', val=50.0,  units='N')
        self.add_output('W_empty', val=15.0, units='N')

    def setup_partials(self):
        self.declare_partials('W_empty', ['P_inst', 'r', 'W_total'])

    def compute(self, inputs, outputs):
        Pi = inputs['P_inst'][0]
        r  = inputs['r'][0]
        W  = inputs['W_total'][0]

        W_motor = (K_MOTOR + K_ESC) * Pi * G               # Eq. 4+5
        W_rotor = N_ROTOR * (K_ROTOR_A * r**2 - K_ROTOR_B * r) * G  # Eq. 6 × N
        W_frame = 0.5 * G + BETA_HEX * W                   # Eq. 8 (β=0.20)

        outputs['W_empty'] = W_motor + W_rotor + W_frame

    def compute_partials(self, inputs, partials):
        r  = inputs['r'][0]
        partials['W_empty', 'P_inst']  = (K_MOTOR + K_ESC) * G
        partials['W_empty', 'r']       = N_ROTOR * (2.0 * K_ROTOR_A * r - K_ROTOR_B) * G
        partials['W_empty', 'W_total'] = BETA_HEX


# ── Mission energy ────────────────────────────────────────────────────────

class MissionEnergyComp(om.ExplicitComponent):
    """
    Eq. (9): E = P_hover·(2·(n_c+1)·t_hover) + P_cruise·(R/V∞)
    Identical to QBiT.

    Inputs:  P_hover [W], P_cruise [W], V_inf [m/s], R [m], n_c
    Output:  E_req [J]
    """

    def setup(self):
        self.add_input('P_hover',  val=400.0, units='W')
        self.add_input('P_cruise', val=250.0, units='W')
        self.add_input('V_inf',    val=18.0,  units='m/s')
        self.add_input('R',        val=30e3,  units='m')
        self.add_input('n_c',      val=1.0)
        self.add_output('E_req',   val=5e5,   units='J')

    def setup_partials(self):
        self.declare_partials('E_req', ['P_hover', 'P_cruise', 'V_inf', 'R', 'n_c'])

    def compute(self, inputs, outputs):
        Ph = inputs['P_hover'][0]
        Pc = inputs['P_cruise'][0]
        V  = inputs['V_inf'][0]
        R  = inputs['R'][0]
        nc = int(round(float(inputs['n_c'][0])))

        outputs['E_req'] = Ph * (2.0 * (nc + 1) * T_HOVER) + Pc * (R / V)

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
    Identical to QBiT.

    Input:  E_req [J]
    Output: W_battery [N]
    """

    def setup(self):
        self.add_input('E_req',      val=5e5,  units='J')
        self.add_output('W_battery', val=10.0, units='N')

    def setup_partials(self):
        self.declare_partials('W_battery', 'E_req',
                              val=G / (3600.0 * BATTERY_EFF * BATTERY_DENSITY))

    def compute(self, inputs, outputs):
        outputs['W_battery'] = (inputs['E_req'][0] / 3600.0) / (BATTERY_EFF * BATTERY_DENSITY) * G


# ── Weight residual (Eq. 2) ───────────────────────────────────────────────

class WeightResidualComp(om.ExplicitComponent):
    """
    weight_residual = W_total − W_payload − W_battery − W_empty

    Registered as an equality constraint (= 0) in the optimizer.
    Identical structure to QBiT.

    Inputs:  W_total [N], W_payload [N], W_battery [N], W_empty [N]
    Output:  weight_residual [N]
    """

    def setup(self):
        self.add_input('W_total',          val=50.0,  units='N')
        self.add_input('W_payload',        val=29.43, units='N')
        self.add_input('W_battery',        val=10.0,  units='N')
        self.add_input('W_empty',          val=15.0,  units='N')
        self.add_output('weight_residual', val=0.0,   units='N')

    def setup_partials(self):
        self.declare_partials('weight_residual', 'W_total',   val= 1.0)
        self.declare_partials('weight_residual', 'W_payload', val=-1.0)
        self.declare_partials('weight_residual', 'W_battery', val=-1.0)
        self.declare_partials('weight_residual', 'W_empty',   val=-1.0)

    def compute(self, inputs, outputs):
        outputs['weight_residual'] = (inputs['W_total']
                                      - inputs['W_payload']
                                      - inputs['W_battery']
                                      - inputs['W_empty'])
