"""
QBiTModel - top-level OpenMDAO Group for the QBiT sizing problem.

Hierarchy:
  QBiTModel
    ivc          IndepVarComp  - mission parameters + design variable IVs
    physics      PhysicsGroup  - all sizing physics + weight closure solver
    constraints  ConstraintsGroup - constraint output computations

All internal connections are made via promoted names ('*').
The parent Problem registers design vars, constraints, and objective.
"""
import openmdao.api as om

from qbit.groups.physics_group     import PhysicsGroup
from qbit.groups.constraints_group import ConstraintsGroup


def build_qbit_model(payload_kg: float,
                     range_m:    float,
                     n_c:        int = 1) -> om.Group:
    """
    Return a fully configured QBiT model Group.

    Parameters
    ----------
    payload_kg : total package mass [kg]
    range_m    : one-way depot-to-customer distance [m]
    n_c        : number of customers on the route
    """
    model = om.Group()

    # ── Mission inputs + design variable initial values ────────────────
    ivc = om.IndepVarComp()
    ivc.add_output('W_payload', val=payload_kg * 9.81, units='N',
                   desc='Payload weight (fixed)')
    ivc.add_output('R',  val=2.0 * range_m, units='m',
                   desc='Total route range (round-trip)')
    ivc.add_output('n_c', val=float(n_c),
                   desc='Number of delivery customers (fixed)')

    # Design variable initial guesses (W_total is NOT here — it is the
    # implicit state of WeightBalanceComp; adding it here would create two
    # outputs for the same promoted name → singular Jacobian)
    ivc.add_output('V_inf', val=33.0,       units='m/s')
    ivc.add_output('r',     val=0.22,       units='m')
    ivc.add_output('J',     val=1.3,        desc='Propeller advance ratio')
    ivc.add_output('S_w',   val=0.20,       units='m**2')

    model.add_subsystem('ivc',         ivc,               promotes=['*'])
    model.add_subsystem('physics',     PhysicsGroup(),    promotes=['*'])
    model.add_subsystem('constraints', ConstraintsGroup(), promotes=['*'])

    return model
