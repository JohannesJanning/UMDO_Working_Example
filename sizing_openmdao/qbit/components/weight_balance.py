"""
WeightResidualComp - Eq. (2) of Kaneko & Martins (2023).

Explicit weight closure residual:
  weight_residual = W_total - W_payload - W_battery - W_empty

Registered as an EQUALITY CONSTRAINT (= 0) in the optimizer.
This is numerically equivalent to the Newton inner loop used in the paper:
the optimizer enforces weight closure at every converged solution.

Using an equality constraint rather than an ImplicitComponent + Newton avoids
complex-step / Newton incompatibility when SLSQP computes total derivatives.
"""
import openmdao.api as om


class WeightResidualComp(om.ExplicitComponent):
    """
    Inputs:  W_total [N], W_payload [N], W_battery [N], W_empty [N]
    Output:  weight_residual [N]  (constrained to = 0 by optimizer)
    """

    def setup(self):
        self.add_input('W_total',        val=60.0,  units='N')
        self.add_input('W_payload',      val=29.43, units='N')
        self.add_input('W_battery',      val=10.0,  units='N')
        self.add_input('W_empty',        val=20.0,  units='N')
        self.add_output('weight_residual', val=0.0, units='N',
                        desc='W_total - W_payload - W_battery - W_empty (= 0 at solution)')

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
