import openmdao.api as om
from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup

yaml_content = """
Variable 0:
  name: t_hover
  distribution: lognormal
  mu: 4.0
  stdev: 0.3
  interval_low: 25.0
  type: aleatory

Settings:
  order: 3
  backend: Agg
  track_convergence_off: False 
  aleat_samp_size: 1000
"""
with open("test_pce.yaml", "w") as f:
    f.write(yaml_content)

try:
    ret = interface.initialize('test_pce.yaml', 'test_run.dat')
    print("Initialize returned:", len(ret))
    print("run_matrix:\n", ret[-1])
except Exception as e:
    import traceback
    traceback.print_exc()
