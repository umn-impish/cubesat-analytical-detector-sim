import os
import sys
sys.path.append('..')

import numpy as np
from HafxSimulationContainer import HafxSimulationContainer
from pull_consts import from_dir

files = os.listdir(from_dir)
classez = ('C1', 'M1', 'M5', 'X1')
files = [
    os.path.join(from_dir, f) for f in files
    if any(cl in f for cl in classez)]
containers = [HafxSimulationContainer.from_saved_file(fn) for fn in files]

total_area = 0
areaz = dict()
start_vec = 43 / 4 * np.ones_like(containers[0].flare_spectrum.energies)
for c in containers:
    eng = c.flare_spectrum.energies
    crit = np.logical_and(eng >= 1, eng <= 300)
    eng = eng[crit]

    effective_area = (start_vec @ c.matrices[c.KPURE_RESPONSE])[crit]
    this_area = np.trapz(x=eng, y=effective_area)
    this_area /= (np.max(eng) - np.min(eng))
    print(f'{c.goes_class} effective area: {this_area:.3f} cm2')
    total_area += this_area

print(f'Average effective area per detector: {total_area / 4:.3f} cm2')
print(f'Total average effective area: {total_area:.3f} cm2')
