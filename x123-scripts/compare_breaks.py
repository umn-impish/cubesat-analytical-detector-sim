import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append('..')

from adetsim.sim_src import FlareSpectrum

breaks = (0, 10, 20, 30)
for gc in ('C1', 'M1', 'X1'):
    fig, axs = plt.subplots(nrows=len(breaks), ncols=1, figsize=(8, 3*len(breaks)))
    axs = axs.flatten()

    fig.suptitle(f'{gc} class')
    for be, ax in zip(breaks, axs):
        fs = FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc, start_energy=2, end_energy=300, de=0.01, break_energy=be)

        ax.plot(fs.energies, fs.thermal, label='thermal', color='r')
        ax.plot(fs.energies, fs.nonthermal, label='nonthermal', color='k')
        ax.plot(fs.energies, fs.flare, label='sum', color='blue')
        m = np.max(fs.flare)
        ax.axhline(y=m, label=f'max flux = {m:.2e}', color='gray', alpha=0.4, linestyle=':')
        ax.set_ylabel('photons / sec / cm${}^2$ / keV')
        ax.set_title(f'spectral index break energy @ {be} keV')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-4)
        ax.legend()

    axs[-1].set_xlabel('energy (keV)')
    fig.tight_layout()
    fig.savefig(f'compare-spec-{gc}.pdf')
# plt.show()
