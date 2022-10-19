import os

import numpy as np
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from pull_consts import from_dir

files = os.listdir(from_dir)
classez = ('C1', 'M1', 'M5', 'X1')
files = [
    os.path.join(from_dir, f) for f in files
    if any(cl in f for cl in classez)]
print(files)
containers = [HafxSimulationContainer.from_saved_file(fn) for fn in files]

total_area = 0
areaz = dict()
start_vec = 43 / 4 * np.ones_like(containers[0].flare_spectrum.flare)
for c in containers:
    edges = c.flare_spectrum.energy_edges
    crit = np.logical_and(edges >= 1, edges <= 300)
    filtered_edges = edges[crit]

    effective_area = c.matrices[c.KPURE_RESPONSE] @ start_vec
    this_area = np.sum(effective_area * np.diff(edges)[crit[:-1]])
    this_area /= (np.max(filtered_edges) - np.min(filtered_edges))
    print(f'{c.goes_class} effective area: {this_area:.3f} cm2')
    total_area += this_area

print(f'Average effective area per detector: {total_area / 4:.3f} cm2')
print(f'Total average effective area: {total_area:.3f} cm2')