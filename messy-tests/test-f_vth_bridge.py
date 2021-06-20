import os
import sys
sys.path.append('..')
import numpy as np
from sswidl_bridge import f_vth_bridge

emin = 1
emax = 300
de = 0.1
evec = np.arange(emin, emax, step = de)
thermal_spec = f_vth_bridge(evec, 0.20340744608778885, 19.563430168635527, 1.0)
print(f"Thermal size: {thermal_spec.size}", f"Evec size: {evec.size}", sep=os.linesep)
