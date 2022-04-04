import lzma
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys; sys.path.append('..')

from adetsim.hafx_src import HafxSimulationContainer as HSC

# plt.style.use('/Users/settwi/mpl-styles/fig.mplstyle')

saved_hafx_dir = '../cebr3-scripts/misc-scripts/optimized-2021-dec-24'
flare_focus = ['C1', 'C5', 'M5', 'X1']

keepme_hafx_files = [fn for fn in os.listdir(saved_hafx_dir) if any(c in fn for c in flare_focus)]

hafx_containerz = [
    HSC.HafxSimulationContainer.from_file(os.path.join(saved_hafx_dir, fn)) for fn in keepme_hafx_files]

with lzma.open(next(fn for fn in os.listdir() if 'xz' in fn), 'rb') as f:
    x123_dat = pickle.load(f)

fig, ax = plt.subplots()#figsize=(8, 6))

ax.plot(
    x123_dat['energies'],
    x123_dat['area'] * x123_dat['undisp_resp'] @ np.ones_like(x123_dat['energies']),
    label='X-123')

for c in hafx_containerz:
    ax.plot(
        c.flare_spectrum.energies,
        c.compute_effective_area(),
        label=f'{c.goes_class}-optimized scintillator')

ax.axvline(x=20, label='20 keV boundary', linestyle='--', alpha=0.5, color='black')
# ax.axhline(y=43/4, label=f'Scintillator channel geometric area ({43/4:.2f} cm${{}}^2$)',
#            color='k', alpha=0.5, linestyle='-.')
# ax.axhline(y=0.25, label='X-123 geometric area (0.25 cm${}^2$)', color='k', alpha=0.5, linestyle=':')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Effective area (cm${}^2$)')
ax.set_title('Effective area comparison')

ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(0, 300)
ax.set_ylim(0.01, 13)
ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    bottom=True, top=True, left=True, right=True,
)
ax.minorticks_on()
ax.legend()

fig.tight_layout()
# fig.savefig('compare-hafx-x123-effective-area.pdf')
plt.show()
