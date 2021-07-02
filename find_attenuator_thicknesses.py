import os
import numpy as np
from scipy.integrate import simpson

import common
import sim_src.impress_constants as ic
from sim_src.FlareSpectrum import FlareSpectrum


def save_simulated(sim_dict, gcls, pfx=''):
    save_fname = f"{pfx}{gcls}_{sim_dict[ic.THICK_KEY]:.3e}_energies-flare-response-area-thickness"
    np.savez_compressed(os.path.join(ic.DATA_DIR, save_fname), **sim_dict)
    common.print_log(f"Saved {save_fname}")


def gen_sim_quants(ds, fs, thick):
    ds.materials[0].thickness = thick
    ret = dict()
    ret[ic.ENG_KEY] = fs.energies
    ret[ic.UNDISP_KEY] = ds.generate_detector_response_to(fs, False)
    avec = np.ones(fs.energies.size) * ic.SINGLE_DET_AREA
    ret[ic.EFFA_KEY] = np.matmul(ret[ic.UNDISP_KEY], avec)
    ret[ic.RESP_KEY] = ds.apply_detector_dispersion_for(fs, ret[ic.UNDISP_KEY])
    return ret


def flare_spectra_iter(gcz):
    for gc in gcz:
        yield FlareSpectrum.make_with_battaglia_scaling(gc, ic.E_MIN, ic.E_MAX, ic.DE)


def count_edge(cts, target, dt):
    '''
    when we want to in the thickness-finding loop.
        with positive increment, cts < target, i.e. too much attenuation
        with negative increment, cts > target, i.e. attenuator is too thin
    '''
    if dt > 0: return cts < target
    elif dt < 0: return cts > target
    else: raise ValueError("dt indistinguishable from zero")


def appr_count_step(ds, fs, thick, target_cps):
    '''
    start with thickness that's sure to attenuate the flare
    zigzag around target count rate until we're close enough for gov't work (below and within 5%)
    '''
    eng = fs.energies
    step = -1 * thick / 10
    divs = 0
    TOL = 0.05
    MAX_DIVS = 8
    restrict = np.logical_and(eng >= ic.E_TH_MIN, eng <= ic.E_TH_MAX)

    while divs < MAX_DIVS and thick > 0:
        common.print_log(f"{fs.goes_class}: {thick:.4e} cm")
        the_goods = gen_sim_quants(ds, fs, thick)
        counts_per_kev = np.matmul(the_goods[ic.RESP_KEY], fs.flare) * ic.SINGLE_DET_AREA
        cur_counts = simpson(counts_per_kev[restrict], x=eng[restrict])
        if count_edge(cur_counts, target_cps, step):
            common.print_log("Found the count edge.\n", f"Counts: {cur_counts}, thickness: {thick:.4e} cm")
            step /= -10
            divs += 1
        thick += step
        delta = 1 - cur_counts/target_cps
        if abs(delta) < TOL and delta > 0:
            break

    if divs == MAX_DIVS:
        common.print_log("** Hit max number of step divisions.")
    if thick < 0:
        common.print_log("** zero attenuator window thickness! uh oh")
    # go back a step and cut off precision
    clean_thick = thick - step
    if clean_thick < 1e-6: clean_thick = 0
    the_goods[ic.THICK_KEY] = clean_thick
    the_goods[ic.FS_KEY] = fs.flare
    return the_goods


def find_appropriate_counts(goes_classes, initial_thickness, target_cps):
    ''' optimize attenuator window for target_cps given various GOES flare sizes '''
    detector_stack = common.generate_impress_stack(ic.HAFX_MATERIAL_ORDER)
    for fs in flare_spectra_iter(goes_classes):
        to_save = appr_count_step(detector_stack, fs, initial_thickness, target_cps)
        save_simulated(to_save, fs.goes_class, 'optimized_')


if __name__ == '__main__':
    common.setup_structure('find_thick')
    classes = ('C1', 'C5', 'M1', 'M5', 'X1')
    init_thick = 0.1
    target_cps = -np.log(0.95) / ic.HAFX_DEAD_TIME
    find_appropriate_counts(classes, init_thick, target_cps)
