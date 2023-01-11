import os

import numpy as np
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from pull_consts import from_dir

LOW_ENERGY_CUT = 30
HIGH_ENERGY_CUT = 100
apply_cut = lambda a: (a > LOW_ENERGY_CUT) & (a < HIGH_ENERGY_CUT)

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
    midpoints = edges[:-1] + np.diff(edges)/2
    mid_crit = apply_cut(midpoints)
    filtered_edges = edges[apply_cut(edges)]

    effective_area = (c.matrices[c.KPURE_RESPONSE] @ start_vec)
    this_area = np.sum((effective_area * np.diff(edges))[mid_crit])
    this_area /= (np.max(filtered_edges) - np.min(filtered_edges))

    print(f'{c.goes_class} average effective area: {this_area:.3f} cm2')
    total_area += this_area

print('Energy range:', LOW_ENERGY_CUT, ',', HIGH_ENERGY_CUT, 'keV')
print(f'Average effective area per detector: {total_area / 4:.3f} cm2')
print(f'Total average effective area: {total_area:.3f} cm2')
