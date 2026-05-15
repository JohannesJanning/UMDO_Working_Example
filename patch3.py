with open("sizing_openmdao/run_qbit_uqpce.py", "r") as f:
    text = f.read()

replacement = """        try:
            pce.confidence_interval()
        except Exception:
            pass"""
text = text.replace("        ", replacement, 1) # Note: we just put it back

# actually let's just write a proper clean fix
with open("sizing_openmdao/run_qbit_uqpce.py", "w") as f:
    pass
