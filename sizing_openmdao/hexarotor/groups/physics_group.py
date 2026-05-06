"""
PhysicsGroup - hexarotor sizing physics.

All components in one group, promoted to '*'.
No group-level Newton solver (weight balance enforced as optimizer equality
constraint, same approach as QBiT and as stated in paper Sec. III.C).

β and λ are converged via internal Newton loops inside AerodynamicTrimComp.
"""
import openmdao.api as om

from hexarotor.components.hover_power  import HoverPowerComp
from hexarotor.components.aero_trim    import AerodynamicTrimComp
from hexarotor.components.cruise_power import CruisePowerComp
from hexarotor.components.sizing_comps import (InstalledPowerComp,
                                                EmptyWeightComp,
                                                MissionEnergyComp,
                                                BatteryWeightComp,
                                                WeightResidualComp)


class PhysicsGroup(om.Group):
    """All hexarotor sizing physics."""

    def setup(self):
        self.add_subsystem('hover',     HoverPowerComp(),      promotes=['*'])
        self.add_subsystem('trim',      AerodynamicTrimComp(), promotes=['*'])
        self.add_subsystem('cruise',    CruisePowerComp(),     promotes=['*'])
        self.add_subsystem('installed', InstalledPowerComp(),  promotes=['*'])
        self.add_subsystem('empty',     EmptyWeightComp(),     promotes=['*'])
        self.add_subsystem('energy',    MissionEnergyComp(),   promotes=['*'])
        self.add_subsystem('battery',   BatteryWeightComp(),   promotes=['*'])
        self.add_subsystem('balance',   WeightResidualComp(),  promotes=['*'])
