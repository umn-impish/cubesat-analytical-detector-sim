import os
import numpy as np
from scipy.integrate import simpson

import common
from HafxSimulationContainer import HafxSimulationContainer
import sim_src.impress_constants as ic
from sim_src.FlareSpectrum import FlareSpectrum, battaglia_iter


def count_edge(cts, target, dt):
    '''
    when we want to in the thickness-finding loop.
        with positive increment, cts < target, i.e. too much attenuation
        with negative increment, cts > target, i.e. attenuator is too thin
    '''
    if dt > 0: return cts < target
    elif dt < 0: return cts > target
    else: raise ValueError("dt indistinguishable from zero")


def appr_count_step(sim_con, target_cps):
    '''
    start with thickness that's sure to attenuate the flare
    zigzag around target count rate until we're close enough for gov't work (below and within 5%)
    '''
    eng = sim_con.flare_spectrum.energies
    step = -1 * sim_con.al_thick / 10
    divs = 0
    TOL = 0.05
    MAX_DIVS = 8
    restrict = np.logical_and(eng >= ic.MIN_THRESHOLD_ENG, eng <= ic.MAX_THRESHOLD_ENG)

    while divs < MAX_DIVS and sim_con.al_thick > 0:
        print(f"{sim_con.flare_spectrum.goes_class}: {sim_con.al_thick:.4e} cm")
        sim_con.simulate()
        counts_per_kev = np.matmul(sim_con.matrices[sim_con.KDISPERSED_RESPONSE], sim_con.flare_spectrum.flare) * ic.SINGLE_DET_AREA
        cur_counts = simpson(counts_per_kev[restrict], x=eng[restrict])
        if count_edge(cur_counts, target_cps, step):
            print("Found the count edge.\n", f"Counts: {cur_counts}, thickness: {sim_con.al_thick:.4e} cm")
            step /= -10
            divs += 1
        sim_con.al_thick = sim_con.al_thick + step
        delta = 1 - cur_counts/target_cps
        if abs(delta) < TOL and delta > 0:
            break

    if divs == MAX_DIVS:
        print("** Hit max number of step divisions.")
    if sim_con.al_thick < 0:
        print("** zero attenuator window thickness! uh oh")
    # go back a step and cut off precision at 1e-6 cm
    clean_thick = sim_con.al_thick - step
    if clean_thick < 1e-6: clean_thick = 0
    sim_con.al_thick = clean_thick


def find_appropriate_counts(goes_classes, initial_thickness, target_cps):
    ''' optimize attenuator window for target_cps given various GOES flare sizes '''
    for fs in battaglia_iter(goes_classes):
        sim_container = HafxSimulationContainer(
                aluminum_thickness=initial_thickness, flare_spectrum=fs)
        # populates matrices of detector_stack
        appr_count_step(sim_container, target_cps)
        sim_container.save_to_file(prefix='optimized')
        print(f"Saved {sim_container.gen_file_name('optimized')}.")


if __name__ == '__main__':
    classes = ('C1', 'C5', 'M1', 'M5', 'X1')
    init_thick = 0.1    # cm
    target_cps = -np.log(0.95) / ic.HAFX_DEAD_TIME
    find_appropriate_counts(classes, init_thick, target_cps)
