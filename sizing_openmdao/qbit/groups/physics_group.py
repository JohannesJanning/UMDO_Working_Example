"""
PhysicsGroup - all sizing physics components in one OpenMDAO Group.
- Imports each individual physics component (each is a separate file with its own calculation logic)
- Creates a new OpenMDAO Group that will contain all physics components (class)
- Defines what components belong to this group (setup())
- Adds one physics component to the assembly line (promotes=['*'] to share variables across components)

"""
import openmdao.api as om

from qbit.components.hover_power   import HoverPowerComp
from qbit.components.aero_trim     import AerodynamicTrimComp
from qbit.components.cruise_power  import CruisePowerComp
from qbit.components.sizing_comps  import (InstalledPowerComp,
                                            EmptyWeightComp,
                                            MissionEnergyComp,
                                            BatteryWeightComp)
from qbit.components.weight_balance import WeightResidualComp


class PhysicsGroup(om.Group):
    """
    Sizing physics group. Weight balance enforced as SLSQP equality constraint.
    """

    def setup(self):
        self.add_subsystem('hover',     HoverPowerComp(),      promotes=['*'])
        self.add_subsystem('trim',      AerodynamicTrimComp(), promotes=['*'])
        self.add_subsystem('cruise',    CruisePowerComp(),     promotes=['*'])
        self.add_subsystem('installed', InstalledPowerComp(),  promotes=['*'])
        self.add_subsystem('empty',     EmptyWeightComp(),     promotes=['*'])
        self.add_subsystem('energy',    MissionEnergyComp(),   promotes=['*'])
        self.add_subsystem('battery',   BatteryWeightComp(),   promotes=['*'])
        self.add_subsystem('balance',   WeightResidualComp(),  promotes=['*'])
