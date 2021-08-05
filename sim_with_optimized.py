import os
import numpy as np
import matplotlib.pyplot as plt

from HafxSimulationContainer import HafxSimulationContainer
from HafxStack import HAFX_DEAD_TIME, FULL_AREA
import sim_src.impress_constants as ic


fig_dir = 'figures'
optim_dir = 'optimized-2-aug-2021'
files = os.listdir(optim_dir)
thresh_counts = -1 * np.log(0.95) / HAFX_DEAD_TIME
chosen_ones = ('C1', 'M1', 'M5', 'X1')


containers = list()
for ch in chosen_ones:
    f = next(fn for fn in files if ch in fn)
    cur = HafxSimulationContainer.from_saved_file(os.path.join(optim_dir, f))
    containers.append(cur)
    print(f"loaded {f}")

flare_spectra = { con.flare_spectrum.goes_class: con.flare_spectrum for con in containers }
original_spectra = { id(c): c.flare_spectrum.goes_class for c in containers }

fig, axs = plt.subplots(2, len(chosen_ones))
for i, simmed in enumerate(chosen_ones):
    # print(f"plotting {simmed}")
    # set the flare spectrum to be the current one for each simulation container
    for c in containers:
        c.flare_spectrum = flare_spectra[simmed]
    energies, flare = flare_spectra[simmed].energies, flare_spectra[simmed].flare

    ax = axs[0, i]
    all_areas = np.array(
        list(con.compute_effective_area(cps_threshold=thresh_counts) for con in containers)
    )
    total_area = np.sum(all_areas, axis=0)
    print(f"Max total effective area: {np.max(total_area)} cm2")

    ax.set_ylim(0, 44)
    ax.set_ylabel("Effective area (cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_title(f"Effective area for optimized detectors, {simmed} flare\n(rejecting detectors with > {int(thresh_counts)} counts/sec)")
    ax.set_xscale('log')
    ax.set_yscale('linear')

    tefa_kwargs = {'label': "Total effective area"} if i == 0 else {}
    fula_kwargs = {'label': "Total (true) detector area"} if i == 0 else {}
    ax.plot(energies, total_area, **tefa_kwargs)
    ax.axhline(y=FULL_AREA, linestyle='--', **fula_kwargs)
    for j, a in enumerate(all_areas):
        gc = original_spectra[id(containers[j])]
        al_thick = containers[j].al_thick
        kwargz = dict()
        if i == 0:
            kwargz['label'] = f"{gc}, {al_thick:.1e}cm Al eff. area"
        ax.plot(energies, a, **kwargz)
    ax.legend()

    ax = axs[1, i]
    ax.set_ylim(1e-4, 1e9)
    ax.set_title(f"Optimized attenuator windows, {simmed} flare")
    ax.set_ylabel("Flare spectrum (counts / keV / s / cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(energies, flare, label=f"Original {simmed} spectrum")
    for c in containers:
        lab = original_spectra[id(c)]
        att_spec = np.matmul(c.matrices[c.KDISPERSED_RESPONSE], flare)
        ax.plot(energies, att_spec, label=f"{lab} optimized, {c.al_thick:.1e}cm Al")
    ax.legend()

# restore the changed flare spectra
for c in containers:
    c.flare_spectrum = flare_spectra[original_spectra[id(c)]]

fig.set_size_inches(24, 9.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'big-boi.pdf'))
