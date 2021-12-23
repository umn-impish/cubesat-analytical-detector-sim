import os
import numpy as np
import matplotlib.pyplot as plt

from HafxSimulationContainer import HafxSimulationContainer
from HafxStack import HAFX_DEAD_TIME, FULL_AREA
import sim_src.impress_constants as ic

try:
    import traceback
    plt.style.use('/Users/sette/agu.mplstyle')
except Exception as e:
    traceback.print_exc()
    input("any key to continue")
    raise

fig_dir = 'figures'
optim_dir = 'optimized-30-nov-2021'
files = os.listdir(optim_dir)
chosen_ones = ('C1', 'M1', 'M5', 'X1')

DEFAULT_THRESH_COUNTS = -1 * np.log(0.95) / HAFX_DEAD_TIME
def plot_effective_areas(ax, sim_cons, cur_flare, include_legend=True, thresh_counts=DEFAULT_THRESH_COUNTS):
    all_areas = np.array(list(
        con.compute_effective_area(
            cps_threshold=thresh_counts, different_flare=cur_flare) for con in sim_cons))
    total_area = np.sum(all_areas, axis=0)
    print(f"Max total effective area: {np.max(total_area)} cm2")

    ax.set_ylim(0, 44)
    ax.set_ylabel("Effective area (cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_title(f"Effective area for optimized detectors, {cur_flare.goes_class} flare\n(rejecting detectors with > {int(thresh_counts)} counts/sec)")
    ax.set_xscale('log')
    ax.set_yscale('linear')

    full_area_kwargs = {'label': "Geometric area"}
    tot_effa_kwargs  = {'label': "Total effective area"}

    ax.axhline(y=FULL_AREA, linestyle='--', **full_area_kwargs)
    ax.plot(cur_flare.energies, total_area, **tot_effa_kwargs)
    for sc, a in zip(sim_cons, all_areas):
        kwargz = dict()
        kwargz['label'] = f"{sc.goes_class} detector eff. area"
        ax.plot(cur_flare.energies, a, **kwargz)
        if include_legend: ax.legend()

def plot_responses(ax, sim_cons, cur_flare):
    ax.set_ylim(1e-4, 1e9)
    ax.set_title(f"Optimized attenuators, {cur_flare.goes_class}")
    ax.set_ylabel("Flare (cts/keV/s/cm${}^2$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    for c in sim_cons:
        att_spec = np.matmul(c.matrices[c.KDISPERSED_RESPONSE], cur_flare.flare)
        ax.plot(c.flare_spectrum.energies, att_spec, label=f"{c.goes_class} detector; {c.al_thick*1e4:.0f}um Al")
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
        ax.plot(fs.energies, fs.flare, label=f"Original {simmed} spectrum")
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
    area_ax.set_title('Simulated analytical effective areas')

    resp_ax.plot(wanted_flare.energies, wanted_flare.flare, label=f"Original {WANTED_FLARE_CLASS} spectrum")
    plot_responses(resp_ax, containers, wanted_flare)
    resp_ax.set_title(f'Analytical responses for {WANTED_FLARE_CLASS} simulated flare')

    tosave = {'resp': resp_fig, 'area': area_fig}
    for n, fig in tosave.items():
        # fig.set_size_inches(8, 6)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{n}-smol-boi.png'))


if __name__ == '__main__':
    # main()
    agu()
