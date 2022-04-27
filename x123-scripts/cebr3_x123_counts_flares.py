import itertools
import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys; sys.path.append('..')

from adetsim.sim_src import FlareSpectrum
from adetsim.hafx_src import HafxSimulationContainer as HSC
from adetsim.hafx_src import X123Stack, X123CdTeStack
from adetsim.hafx_src.HafxStack import SINGLE_DET_AREA as CEBR3_DETECTOR_AREA

from make_x123_response import X123_MAT, build_x123_data

ALL_FN = f'../responses-and-areas/cmp-cebr3-x123-{X123_MAT}.lzma'
SAVED_HAFX_DIR = '../responses-and-areas/optimized-2021-dec-24'
FLARES_SIMULATE = ['C1', 'C5', 'M1', 'M5', 'X1']

def build_all_data():
    loaded = False
    try:
        with lzma.open(ALL_FN, 'rb') as f:
            print('unpickle all data')
            all_responses = pickle.load(f)
            good = all(kk in all_responses.keys() for kk in FLARES_SIMULATE)
            for both_per_flare in all_responses.values():
                good = good and all(kk in both_per_flare['cebr3_response'].keys() for kk in FLARES_SIMULATE)
            if good:
                print('got all the flares we want already')
                return all_responses
            loaded = True
    except FileNotFoundError:
        all_responses = dict()

    keepme_hafx_files = [fn for fn in os.listdir(SAVED_HAFX_DIR) if any(c in fn for c in FLARES_SIMULATE)]
    hafx_containerz = [
        HSC.HafxSimulationContainer.from_file(
            os.path.join(SAVED_HAFX_DIR, fn)
        ) for fn in keepme_hafx_files
    ]
    for goes_class in FLARES_SIMULATE:
        if goes_class in all_responses.keys() and\
                all(
                    kk in all_responses[goes_class]['cebr3_response'].keys()
                    for kk in FLARES_SIMULATE):
            print('good on', goes_class)
            continue

        if loaded: print('patching saved data for', goes_class, 'incident')
        else: print(goes_class, 'from scratch')

        try:
            with lzma.open(X123_BASENAME.format(goes_class), 'rb') as f:
                print('unpickle x123 for', goes_class, 'incident')
                x123_data = pickle.load(f)
        except FileNotFoundError:
            x123_data = build_x123_data(goes_class)

        cebr3_resp = dict()
        fs_patch = FlareSpectrum.FlareSpectrum(
            goes_class=goes_class,
            energies=x123_data['energies'],
            thermal=x123_data['flare'],
            nonthermal=0)

        for c in hafx_containerz:
            print('start optimized cebr3', c.goes_class, 'for', goes_class, 'incident')
            # update and re-simulate
            c.simulate(other_spectrum=fs_patch)
            cebr3_resp[c.goes_class] = c.matrices[c.KDISPERSED_RESPONSE] @ x123_data['flare']
            print('done simulating optimized cebr3', c.goes_class, 'for', goes_class, 'incident')

        all_responses[goes_class] = {
            'x123_response': x123_data,
            'cebr3_response': cebr3_resp
        }
        print('done all responses', goes_class, 'incident')

    print('pickling!')
    with lzma.open(ALL_FN, 'wb') as f:
        pickle.dump(all_responses, f)
    return all_responses


def plot_compare_x123_cebr3(data_dict, coll_time=1, attenuators=('X1',)):
    base_colorz = ['black', 'red', 'blue', 'green', 'purple', 'orange']
    cebr3_cutoff_energy = 20
    x123_cts_cutoff = 1
    for gc in FLARES_SIMULATE:
        ccycle = itertools.cycle(base_colorz)
        fig, ax = plt.subplots(figsize=(8, 6))

        all_resp = data_dict[gc]
        x123_dat = all_resp['x123_response']
        energies = x123_dat['energies']
        cebr3_responses = all_resp['cebr3_response']
        de = np.diff(energies)[0]

        total_area = 4 * CEBR3_DETECTOR_AREA + x123_dat['area']
        ax.plot(energies, x123_dat['flare'] * coll_time * total_area, label='Simulated photon flux', c=next(ccycle), linewidth=2)

        x123_yvals = coll_time * x123_dat['disp'] * x123_dat['area'] * de
        ax.plot(energies, x123_yvals, label=f'X-123 ({"CdTe" if (X123_MAT == "cdte") else "Si"})', c=next(ccycle), linewidth=2)

        keyz = list(sorted(cebr3_responses.keys()))
        for opt in keyz:
            if opt in attenuators:
                resp = cebr3_responses[opt]
                bef_crit = energies <= cebr3_cutoff_energy
                # convert to just counts
                before = resp[bef_crit] * CEBR3_DETECTOR_AREA * de * coll_time
                after = resp[~bef_crit] * CEBR3_DETECTOR_AREA * de * coll_time
                col = next(ccycle)
                ax.plot(energies[bef_crit], before, c=col, alpha=0.3, linewidth=2)
                ax.plot(energies[~bef_crit], after, label=f'{opt}-optimized scintillator', c=col, linewidth=2)

        # bad region
        start_bad_energy = energies[np.argmin(np.abs(x123_yvals - x123_cts_cutoff))]
        if start_bad_energy < cebr3_cutoff_energy:
            ax.axvspan(
                start_bad_energy, cebr3_cutoff_energy,
                label=rf'Missing region (X-123 $\leq$ {x123_cts_cutoff} count)',
                color='red', alpha=0.2, zorder=-1
            )

        ax.minorticks_on()
        ax.tick_params(
            axis='both',
            which='both',
            direction='inout',
            reset=True,
            bottom=True, top=True, left=True, right=True,
        )
        ax.grid(visible=True, which='both', axis='both', alpha=0.3)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(1, 100)
        ax.set_ylim(1e-1, 8e9)

        ax.legend()
        ax.set_title(f'{gc} GOES class analytical detector response; 1s integration')
        ax.set_xlabel('energy (keV)')
        ax.set_ylabel('average counts')

        fig.tight_layout()
        extension = 'png'
        fig.savefig(f'../figures/{gc}-compare-scint-x123-{X123_MAT}.{extension}', dpi=300)
        # plt.show()


def main():
    dat = build_all_data()
    plot_compare_x123_cebr3(dat, coll_time=1, attenuators=['C1', 'M1', 'M5', 'X1'])

if __name__ == '__main__': main()
