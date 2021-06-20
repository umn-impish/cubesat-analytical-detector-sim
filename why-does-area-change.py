import matplotlib.pyplot as plt
import numpy as np
import os

import common
from AttenuationData import AttenuationType
from FlareSpectrum import FlareSpectrum
import impress_constants as ic

fns = os.listdir(os.path.join(ic.DATA_DIR, 'optimized'))
thicks = dict()
for fn in fns:
    sz, t = fn.split('_')[:2]
    thicks[sz] = float(t)

det_stacks = dict()
for k, thick in thicks.items():
    det_stacks[k] = common.generate_impress_stack(ic.HAFX_MATERIAL_ORDER, al_thick=thick)

# we only want effective area. actual spectrum doesn't matter.
energies = np.arange(ic.E_MIN, ic.E_MAX, step=ic.DE)
dummy_fs = FlareSpectrum.dummy(energies)

rayleigh_only = dict()
phot_only = dict()
avec = np.ones(dummy_fs.energies.size) * ic.SINGLE_DET_AREA
for k, ds in det_stacks.items():
    print(f"calculating eff area for {k}")
    rayleigh_only[k] = np.matmul(
            ds.generate_detector_response_to(dummy_fs, False, [AttenuationType.RAYLEIGH]), avec)
    phot_only[k] = np.matmul(
            ds.generate_detector_response_to(dummy_fs, False, [AttenuationType.PHOTOELECTRIC_ABSORPTION]), avec)

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
plt.savefig(os.path.join(ic.FIG_DIR, 'effa-diff-scattering.pdf'))
