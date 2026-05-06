"""
ConstraintsGroup - hexarotor constraint outputs.

Hexarotor Table 1 constraints:
  disk_loading  = (W_total / N_rotor) / (π·r²)  ≤ 250 N/m²
  blade_loading = CT / σ                          ≤ 0.14

No CL constraint — "CL ≤ 0.6 QBiT only" per Table 1.
"""
import openmdao.api as om
from hexarotor.constants import N_ROTOR, SIGMA


class ConstraintsGroup(om.Group):

    def setup(self):
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

        self.add_subsystem(
            'blade_load',
            om.ExecComp(f'blade_loading = CT / {SIGMA}'),
            promotes=['*'],
        )
