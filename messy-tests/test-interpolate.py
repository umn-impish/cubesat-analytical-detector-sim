import numpy as np
import matplotlib.pyplot as plt
from AttenuationData import AttenuationData, AttenuationType
from FlareSpectrum import FlareSpectrum

energies = np.arange(1, 100, step=0.1)
fs = FlareSpectrum('C1', energies, np.array([]), np.array([]))
ad = AttenuationData.from_nist_file('attenuation-data-files/Al.tab')
new = ad.interpolate_from(fs)

fig, ax = plt.subplots()
ax.plot(energies, new.attenuations[AttenuationType.PHOTOELECTRIC_ABSORPTION])
ax.plot(ad.energies, ad.attenuations[AttenuationType.PHOTOELECTRIC_ABSORPTION])
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlim(1, 300)
ax.set_ylim(1e-2, 5000)
plt.show()
