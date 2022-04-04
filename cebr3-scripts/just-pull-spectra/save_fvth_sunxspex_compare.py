import sys
sys.path.append('..')
from sim_src import FlareSpectrum

import numpy as np
classes = ['C1', 'M5', 'X2']

for c in classes:
    fspec = FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(c, 1.1, 200, 0.05)
    e, t = fspec.energies, fspec.thermal
    np.savetxt(f'compare-thermal-{c}.tab', np.array([e, t]).T)
