import lzma
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from x123_attenuators import out_fn, edges

with lzma.open(out_fn, 'rb') as f:
    dat = pickle.load(f)

fig, axs = plt.subplots(figsize=(14, 6), layout='constrained', nrows=2, ncols=2)
axs = axs.flatten()

integration_time = 20 << u.s
x123_area = (17 << u.mm**2).to(u.cm**2).value
de = np.diff(edges)
for (ax, (goes_class, sub_dat)) in zip(axs, dat.items()):
    for (k, spec) in sub_dat.items():
        ct = spec * x123_area * de * integration_time.value
        ax.stairs(
            ct,
            edges,
            label=f'{k} ({ct.sum():.0f} ct)'
        )

    ax.set(
        xlabel='energy (keV)', ylabel='ct',
        title=f'Al x-123 attenuator | {goes_class}',
        xscale='log', yscale='log',
        ylim=(1, None)
    )
    ax.legend()

fig.suptitle(f'counts in {integration_time} integration')
fig.savefig(f'{integration_time}-integration.png', dpi=300)
plt.show()
