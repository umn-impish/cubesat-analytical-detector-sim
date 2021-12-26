import sys
sys.path.append('..')
from sim_src import FlareSpectrum

import numpy as np

classes = ['C1', 'C5', 'M1', 'M5', 'X1']
fsa = [FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(c, 1, 300, 0.05) for c in classes]

kev_to_mev = 1 / 1000
for fs in fsa:
    chop_lowenergy = fs.energies > 5
    np.savetxt(
        f'raw_{fs.goes_class.lower()}.txt',
        np.transpose([fs.energies[chop_lowenergy], fs.flare[chop_lowenergy]]))
