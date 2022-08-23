import os
import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append(os.getenv('ADETSIM_PATH'))

from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from adetsim.hafx_src.HafxMaterialProperties import HAFX_DEAD_TIME, FULL_AREA
import adetsim.sim_src.impress_constants as ic

sty = os.getenv('MPL_ORAL_STYLE')
if sty is not None:
    plt.style.use(sty)

fig_dir = 'figures'
optim_dir = '../responses-and-areas/optimized-2022-aug-22-bins'
files = os.listdir(optim_dir)
chosen_ones = ('C1', 'M1', 'M5', 'X1')

legend_loc = (0.5, 0.05)

DEFAULT_THRESH_COUNTS = -1 * np.log(0.95) / HAFX_DEAD_TIME
def plot_effective_areas(ax, sim_cons, cur_flare, include_legend=True, thresh_counts=DEFAULT_THRESH_COUNTS):
    all_areas = np.array(list(
        con.compute_effective_area(
            cps_threshold=thresh_counts, different_flare=cur_flare) for con in sim_cons))
    total_area = np.sum(all_areas, axis=0)
    print(f"Max total effective area: {np.max(total_area)} cm2")

    ax.set_ylim(bottom=0.1, top=100)
    ax.set_ylabel("Effective area (cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_title(f"Effective area for optimized detectors, {cur_flare.goes_class} flare\n(rejecting detectors with > {int(thresh_counts)} counts/sec)")
    ax.set_xscale('log')
    ax.set_yscale('log')

    full_area_kwargs = {'label': "Geometric area", 'color': 'black'}
    tot_effa_kwargs  = {'label': "Total effective area"}

    ax.axhline(y=FULL_AREA, linestyle='--', **full_area_kwargs)
    ax.stairs(total_area, cur_flare.energy_edges, **tot_effa_kwargs)
    for sc, a in zip(sim_cons, all_areas):
        kwargz = dict()
        kwargz['label'] = f"{sc.goes_class} detector eff. area"
        ax.stairs(a, cur_flare.energy_edges, **kwargz)
        if include_legend: ax.legend(loc=legend_loc, fontsize=24)


def plot_responses(ax, sim_cons, cur_flare):
    ax.set_ylim(1e-4, 1e9)
    ax.set_title(f"Optimized attenuators, {cur_flare.goes_class}")
    ax.set_ylabel("Flare (cts/keV/s/cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    for c in sim_cons:
        att_spec = np.matmul(c.matrices[c.KDISPERSED_RESPONSE], cur_flare.flare)
        ax.stairs(att_spec, c.flare_spectrum.energy_edges, label=f"{c.goes_class} detector; {c.al_thick*1e4:.0f}um Al")
    ax.legend()


def load_sim_cons():
    conts = list()
    for ch in chosen_ones:
        f = next(fn for fn in files if ch in fn)
        cur = HafxSimulationContainer.from_saved_file(os.path.join(optim_dir, f))
        conts.append(cur)
        print(f"loaded {f}")
    return conts


def main():
    containers = load_sim_cons()
    fig, axs = plt.subplots(2, len(chosen_ones))

    for i, simmed in enumerate(chosen_ones):
        fs = next(c for c in containers if c.goes_class == simmed).flare_spectrum
        try: axs = axs[:, i]
        except IndexError: pass

        plot_effective_areas(axs[0], containers, fs, include_legend=(i == 0))
        ax.stairs(fs.flare, fs.energy_edges, label=f"Original {simmed} spectrum")
        plot_responses(axs[1], containers, fs)

    fig.set_size_inches(24, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'big-boi.png'))


def agu():
    WANTED_FLARE_CLASS = 'M5'
    containers = load_sim_cons()
    wanted_flare = next(
        c for c in containers if c.goes_class == WANTED_FLARE_CLASS).flare_spectrum
    resp_fig, resp_ax = plt.subplots()
    area_fig, area_ax = plt.subplots()

    plot_effective_areas(
        area_ax, containers, wanted_flare,
        thresh_counts=0, include_legend=True)
    area_ax.set_title('Analytical effective areas')

    resp_ax.stairs(wanted_flare.flare, wanted_flare.energy_edges, label=f"Original {WANTED_FLARE_CLASS} spectrum")
    plot_responses(resp_ax, containers, wanted_flare)
    resp_ax.set_title(f'Analytical responses for {WANTED_FLARE_CLASS} simulated flare')

    tosave = {'resp': resp_fig, 'area': area_fig}
    for n, fig in tosave.items():
        # fig.set_size_inches(8, 6)
        fig.tight_layout(pad=0.6)
        fig.savefig(os.path.join(fig_dir, f'{n}-smol-boi.pdf'))


if __name__ == '__main__':
    # main()
    agu()
