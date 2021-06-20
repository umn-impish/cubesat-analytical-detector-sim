import numpy as np
import matplotlib.pyplot as plt
import os

import common
import impress_constants as ic


files = os.listdir(os.path.join(ic.DATA_DIR, 'optimized'))
energies = np.arange(ic.E_MIN, ic.E_MAX, step=ic.DE)

focus = 'M1'
load_in = ('C5', 'M1', 'M5', 'X1')
compute = (focus, )
print("loading flares")
ideal = common.load_optimized_hafx(os.path.join(ic.DATA_DIR, 'optimized'), load_in)
loaded = common.compute_optimized_quantities(ideal, compute, thresh_counts = 0)
effa = loaded['effa'][focus]

fig, ax = plt.subplots()
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Effective area (cm${}^2$)')
ax.set_title("effective areas for various detectors")
ax.set_xscale('log')
ax.set_ylim(0, 11)
for k, ea in effa.items():
    ax.plot(energies, ea, label=k)
ax.legend()
plt.savefig(os.path.join(ic.FIG_DIR, 'effa-comp.pdf'))
