with open("sizing_openmdao/run_qbit_uqpce.py", "r") as f:
    text = f.read()

text = text.replace("cil, cih = pce.confidence_interval()", "")

with open("sizing_openmdao/run_qbit_uqpce.py", "w") as f:
    f.write(text)
