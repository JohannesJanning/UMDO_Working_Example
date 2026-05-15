import openmdao.api as om
from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup

yaml_content = """
Variable 0:
  distribution: lognormal
  mu: 55.0
  stdev: 18.0

UQ:
  order: 2
  dimension: 1
"""
with open("test_pce.yaml", "w") as f:
    f.write(yaml_content)

try:
    ret = interface.initialize('test_pce.yaml', 'test_run.dat')
    print("Initialize returned:", len(ret))
    print("var_basis:", ret[0])
    print("run_matrix:", ret[-1])
except Exception as e:
    import traceback
    traceback.print_exc()
