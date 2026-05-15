with open("sizing_openmdao/run_qbit_uqpce.py", "r") as f:
    text = f.read()

text = text.replace("pce.settings['order'] = int(pce.settings['order'])", "")

with open("sizing_openmdao/run_qbit_uqpce.py", "w") as f:
    f.write(text)
