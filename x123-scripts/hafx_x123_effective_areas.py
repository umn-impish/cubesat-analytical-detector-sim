import lzma
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys; sys.path.append('..')

from adetsim.hafx_src import HafxSimulationContainer as HSC

# plt.style.use('/Users/settwi/mpl-styles/fig.mplstyle')

x123_type = 'cdte'
x123_thk = '1mm' # 500um
x123_fn = f'../responses-and-areas/C5-x123-{x123_type}-resp-{x123_thk}.lzma'
saved_hafx_dir = '../responses-and-areas/optimized-2021-dec-24'
flare_focus = ['C1', 'M5', 'M1', 'X1']

keepme_hafx_files = [fn for fn in os.listdir(saved_hafx_dir) if any(c in fn for c in flare_focus)]

hafx_containerz = [
    HSC.HafxSimulationContainer.from_file(os.path.join(saved_hafx_dir, fn)) for fn in keepme_hafx_files]
hafx_containerz.sort(key=lambda c: c.goes_class)

with lzma.open(x123_fn, 'rb') as f:
    x123_dat = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    x123_dat['energies'],
    x123_dat['area'] * x123_dat['undisp_resp'] @ np.ones_like(x123_dat['energies']),
    label=f'X-123 ({"Si" if x123_type == "si" else "CdTe"})')

for c in hafx_containerz:
    ax.plot(
        c.flare_spectrum.energies,
        c.compute_effective_area(),
        label=f'{c.goes_class}-optimized scintillator')

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
ax.grid(visible=True, which='both', axis='both', alpha=0.3)
ax.legend()

fig.tight_layout()
# fig.savefig('compare-hafx-x123-effective-area.pdf')
fig.savefig(f'../figures/compare-hafx-x123-{x123_type}-effective-area.png', dpi=300)
# plt.show()
