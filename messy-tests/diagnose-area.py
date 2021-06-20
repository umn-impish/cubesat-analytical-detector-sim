import numpy as np
import matplotlib.pyplot as plt

import impress_constants as ic
from FlareSpectrum import FlareSpectrum
from common import generate_impress_stack

fs = FlareSpectrum.make_with_battaglia_scaling('M1', ic.E_MIN, ic.E_MAX, ic.DE)
# thickness is ideal for M1 flare (according to current model)
ds = generate_impress_stack(ic.HAFX_MATERIAL_ORDER, 3e-3)
unsmeared = ds.generate_detector_response_to(fs, False)
avec = np.ones(fs.energies.size) * ds.area
effa = np.matmul(unsmeared, avec)
eng = fs.energies

cleanup = []
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Effective area (cm${}^2$)')
ax.set_title('Effective area shenanigans')
ax.set_xlim(1, 300)
ax.set_ylim(0, 11)
ax.axhline(y=ic.SINGLE_DET_AREA, linestyle='--')
ax.plot(eng, effa)
plt.savefig('ea-diag.pdf')

ax.clear()

ax.set_ylabel('photons / keV / cm2 / s')
ax.set_xlabel('energy (keV)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-4, 1e9)
smeared = ds.apply_detector_dispersion_for(fs, unsmeared)
att = np.matmul(smeared, fs.flare)
ax.plot(eng, att, label="attenuated")
ax.plot(eng, fs.flare, label="original")
ax.set_title("spectrum shenanigans")
plt.savefig("spec-diag.pdf")
