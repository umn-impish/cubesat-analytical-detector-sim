import lzma
import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.insert(0, '..')

import adetsim.sim_src.FlareSpectrum as FS
from adetsim.hafx_src.X123CdTeStack import X123CdTeStack

AMPTEK_FN = 'amptek-data-x123-cdte.tab'

fs = FS.FlareSpectrum.make_with_battaglia_scaling('C1', 1, 500, 0.1)
print('done flare spectrum')

xs = X123CdTeStack()
print('make response matrix')
mtx = xs.generate_detector_response_to(fs, disperse_energy=False)
print('done response matrix')
my_engs = fs.energies
my_prob = mtx @ np.ones_like(my_engs)
print('done multiply prob mine')

am_engs, am_prob = np.loadtxt(AMPTEK_FN, unpack=True)

fig, ax = plt.subplots()

ax.plot(am_engs, am_prob, label='AmpTek provided')
ax.plot(my_engs, my_prob * 100, label='My calculated')

ax.set_title('Compare model and company probabilities, X-123 CdTe')
ax.set_ylabel('Interaction probability (%)')
ax.set_xlabel('Energy (keV)')

fig.tight_layout()
fig.set_size_inches(8, 6)
plt.show()
