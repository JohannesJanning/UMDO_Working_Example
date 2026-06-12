"""
ConstraintsGroup 
- constraint calculation module
"""
import openmdao.api as om
from qbit.constants import N_ROTOR, SIGMA, RHO_AIR



class ConstraintsGroup(om.Group):
    """Collects constraint output computations."""

    def setup(self):
        # Disk loading: T/A = (W_total/N_rotor) / (π·r²)
        self.add_subsystem(
            'disk_load',
            om.ExecComp(
                f'disk_loading = (W_total / {N_ROTOR}) / (3.14159265358979 * r**2)',
                W_total={'units': 'N'},
                r={'units': 'm'},
                disk_loading={'units': 'N/m**2'},
            ),
            promotes=['*'],
        )

        # Blade loading: CT/σ
        self.add_subsystem(
            'blade_load',
            om.ExecComp(f'blade_loading = CT / {SIGMA}'),
            promotes=['*'],
        )

        # Cruise lift coefficient
        self.add_subsystem(
            'lift_coeff',
            om.ExecComp(
                f'cruise_CL = W_total / (0.5 * {RHO_AIR} * V_inf**2 * S_w)',
                W_total={'units': 'N'},
                V_inf={'units': 'm/s'},
                S_w={'units': 'm**2'},
            ),
            promotes=['*'],
        )
