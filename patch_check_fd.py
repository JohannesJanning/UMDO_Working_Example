import numpy as np
import openmdao.api as om
import qbit.components.sizing_comps as sc
from qbit.models.qbit_model import build_qbit_model
from qbit.constants import G, W_TOTAL_BOUNDS
from scipy.optimize import brentq

def inner_solve_for_Wtotal(prob, t, payload, range_m, n_c, dvars):
    V, r, J, Sw = dvars
    
    def eval_res(W):
        op = getattr(sc, 'T_HOVER', None)
        sc.T_HOVER = float(t)
        try:
            prob.set_val('W_total', W)
            prob.set_val('V_inf', V)
            prob.set_val('r', r)
            prob.set_val('J', J)
            prob.set_val('S_w', Sw)
            prob.run_model()
            return float(prob.get_val('weight_residual')[0])
        except Exception:
            return float('nan')
        finally:
            if op is None:
                if hasattr(sc, 'T_HOVER'): del sc.T_HOVER
            else:
                sc.T_HOVER = op

    wl, wh = float(W_TOTAL_BOUNDS[0]), float(W_TOTAL_BOUNDS[1])
    try:
        rl, rh = eval_res(wl), eval_res(wh)
        if rl*rh > 0:
            xs = np.linspace(wl, wh, 9)
            for i in range(8):
                if eval_res(xs[i])*eval_res(xs[i+1]) <= 0:
                    wl, wh = xs[i], xs[i+1]; break
        root = brentq(eval_res, wl, wh, xtol=1e-3, maxiter=50)
        return root
    except Exception:
        return float('nan')

class AsyncQBiTComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('V_inf', val=33.0)
        self.add_output('W_total', shape=(1,))
        self.declare_partials('W_total', 'V_inf', method='fd', step=1e-2)
        
        self.prob = om.Problem(reports=None)
        self.prob.model = build_qbit_model(3.0, 15000.0, 2)
        self.prob.model.set_input_defaults('V_inf', val=33.0, units='m/s')
        self.prob.model.set_input_defaults('r', val=0.22, units='m')
        self.prob.model.set_input_defaults('J', val=1.3)
        self.prob.model.set_input_defaults('S_w', val=0.2, units='m**2')
        self.prob.model.set_input_defaults('W_total', val=6.0 * G, units='N')
        self.prob.setup()

    def compute(self, inputs, outputs):
        dvars = (inputs['V_inf'][0], 0.22, 1.3, 0.2)
        r = inner_solve_for_Wtotal(self.prob, 55.0, 3.0, 15000.0, 2, dvars)
        outputs['W_total'] = [r]

p = om.Problem()
p.model.add_subsystem('comp', AsyncQBiTComp())
p.setup()
p.run_model()
p.check_partials(compact_print=True)
