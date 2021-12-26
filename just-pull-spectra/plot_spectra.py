import sys
sys.path.append('..')
from HafxSimulationContainer import HafxSimulationContainer
from pull_consts import from_dir as opt_dir

import matplotlib.pyplot as plt
import numpy as np
import os

optimized = [os.path.join(opt_dir, fn) for fn in os.listdir(opt_dir)]
containers = [HafxSimulationContainer.from_saved_file(f) for f in optimized]
containers.sort(key=lambda c: c.flare_spectrum.goes_class)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
for c in containers:
    goes_class = c.flare_spectrum.goes_class
    da_matrix = np.identity(c.matrices[c.KDISPERSED_RESPONSE].shape[0])
    fs = c.flare_spectrum
    put_on_yax = da_matrix @ fs.flare
    ax.plot(fs.energies, put_on_yax, label=f'loaded {goes_class} @ {c.al_thick * 1e4:.0f} um Al')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Flux (ph / (cm2 s keV))')
ax.set_title('Look at saved flares')
ax.set_ylim(1e-4)
ax.legend()
fig.tight_layout()
plt.show()
