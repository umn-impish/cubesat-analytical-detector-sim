import matplotlib.pyplot as plt
import numpy as np
import os

from sim_src.AttenuationData import AttenuationType
from sim_src.FlareSpectrum import FlareSpectrum
from HafxSimulationContainer import HafxSimulationContainer
from HafxStack import HafxStack, SINGLE_DET_AREA

data_dir = 'optimized-2-aug-2021'
fig_dir = 'figures'

det_stacks = dict()
for fn in (os.path.join(data_dir, bfn) for bfn in os.listdir(data_dir)):
    sim_con = HafxSimulationContainer.from_saved_file(fn)
    det_stacks[sim_con.flare_spectrum.goes_class] = sim_con.detector_stack

rayleigh_only = dict()
phot_only = dict()
energies = sim_con.flare_spectrum.energies
avec = np.ones(energies.size) * SINGLE_DET_AREA
for k, ds in det_stacks.items():
    print(f"calculating eff area for {k}")
    rayleigh_only[k] = np.matmul(
            ds.generate_detector_response_to(sim_con.flare_spectrum, False, [AttenuationType.RAYLEIGH]), avec)
    phot_only[k] = np.matmul(
            ds.generate_detector_response_to(sim_con.flare_spectrum, False, [AttenuationType.PHOTOELECTRIC_ABSORPTION]), avec)

fig, axs = plt.subplots(1, 2)
for k in rayleigh_only.keys():
    print(f"plotting {k}")
    axs[0].plot(energies, rayleigh_only[k], label="optimized for " + k)
    axs[1].plot(energies, phot_only[k], label="optimized for " + k)

axs[0].set_title('Rayleigh scattering')
axs[1].set_title('Photoelectric absorption')
for ax in axs:
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Effective area (cm${}^2$)')
    ax.set_xscale('log')
    ax.set_ylim(0, 11)
    ax.legend()

fig.set_size_inches(16, 8)
plt.savefig(os.path.join(fig_dir, 'effa-diff-scattering.pdf'))
