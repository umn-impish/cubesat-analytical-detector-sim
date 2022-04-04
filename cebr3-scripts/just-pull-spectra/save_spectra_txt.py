import os
import sys
sys.path.append('..')

from HafxSimulationContainer import HafxSimulationContainer
from sim_src import FlareSpectrum
from pull_consts import from_dir

import numpy as np

files = [os.path.join(from_dir, fn) for fn in os.listdir(from_dir)]
containers = [HafxSimulationContainer.from_saved_file(f) for f in files]
fsa = [c.flare_spectrum for c in containers]
# classes = ['C1', 'C5', 'M1', 'M5', 'X1']
# fsa = [FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(c, 1, 300, 0.05) for c in classes]

kev_to_mev = 1 / 1000
for fs in fsa:
    chop_lowenergy = fs.energies > 0
    np.savetxt(
        f'raw_{fs.goes_class.lower()}.txt',
        np.transpose([fs.energies[chop_lowenergy], fs.flare[chop_lowenergy]]))
