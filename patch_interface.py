from uqpce.pce.io import read_input_file
from uqpce.pce.pce import PCE
import numpy as np

def initialize_patched(input_file, matrix_file):
    var_dict, settings = read_input_file(input_file)
    for k in ['plot', 'verbose']:
        settings.pop(k, None)
    pce = PCE(outputs=False, plot=False, verbose=False, **settings)
    for key, value in var_dict.items():
        pce.add_variable(**value)
    
    # We must generate matrix_file if it doesn't exist
    X = pce.sample(count=10) # say 10
    
    pce.fit(X, X[:, 0]**2) # FIT TO NON_ZERO DUMMY DATA!
    cil, cih = pce.confidence_interval()
    
    return (
        pce._matrix.var_basis_sys_eval, pce._matrix.norm_sq,
        pce._pbox.var_basis_resamp.astype(float), pce._pbox.aleat_samps,
        pce._pbox.epist_samps, X.shape[0], pce.order, pce.variables,
        pce.significance, pce._X
    )
