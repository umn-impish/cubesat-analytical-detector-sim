import os
import numpy as np
import matplotlib.pyplot as plt

import impress_constants as ic
import common


files = os.listdir(ic.DATA_DIR)
# something weird going on here. need to re-run simulation. hmmm
energies = np.arange(ic.E_MIN, ic.E_MAX, step=ic.DE)
thresh_counts = -1 * np.log(0.95) / ic.HAFX_DEAD_TIME
# we need to be able to image an X1 flare.
# we need to have some sort of dynamic range.
chosen_ones = ('C1', 'M1', 'M5', 'X1')

print("loading flares")
ideal = common.load_optimized_hafx(os.path.join(ic.DATA_DIR, 'optimized'), chosen_ones)
loaded = common.compute_optimized_quantities(ideal, chosen_ones, thresh_counts)

fig, axs = plt.subplots(2, len(chosen_ones))
for i, simmed in enumerate(chosen_ones):
    ax = axs[0, i]
    ar_fn = os.path.join(ic.FIG_DIR, f"eff_area_{simmed}_optimized.pdf")
    spec_fn = os.path.join(ic.FIG_DIR, f"spectra_{simmed}_optimized.pdf")
    all_areas = np.array(list(loaded['effa'][simmed].values()))
    total = np.sum(all_areas, axis=0)
    ax.set_ylim(0, 44)
    ax.set_ylabel("effective area (cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_title(f"effective area for optimized detectors, {simmed} flare")
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.plot(energies, total, label="Effective area")
    ax.axhline(y=ic.FULL_AREA, linestyle='--', label="Total detector area")
    ax.legend()

    ax = axs[1, i]
    ax.set_ylim(1e-4, 1e9)
    ax.set_title(f"spectra for optimized detectors, {simmed} flare")
    ax.set_ylabel("flare spectrum (counts / keV / s / cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(energies, loaded['orig'][simmed], label=f"Original {simmed} spectrum")
    for opt, spec in loaded['att'][simmed].items():
        ax.plot(energies, spec, label=f"{opt}")
    ax.legend()

fig.set_size_inches(20, 8)
plt.tight_layout()
plt.savefig(os.path.join(ic.FIG_DIR, 'big-boi.pdf'))
