from adetsim.sim_src import FlareSpectrum
import matplotlib.pyplot as plt
import numpy as np
import sys

goes_class = sys.argv[1]
se, ee, de = 1, 300, 0.05
fs = FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
    goes_class, se, ee, de, break_energy=0)

fig, ax = plt.subplots()
# ax.stairs(fs.flare, fs.energy_edges, label='me')
ax.plot(fs.energies, fs.flare, label='me')
ax.legend()
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Photon flux (count / cm${}^2$ / s / keV)')
ax.set_title(f'Example template {goes_class} flare')
ax.set_xscale('log')
ax.set_yscale('log')
fig.set_size_inches(8, 6)
plt.show()
