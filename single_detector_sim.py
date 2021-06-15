import numpy as np
# import time
import os

import impress_constants as ic
from AttenuationData import AttenuationData
from DetectorStack import HafxStack
from FlareSpectrum import FlareSpectrum
from Material import Material
from PhotonDetector import Sipm3000

def gen_base_stack():
    ''' make the materials in every detector stack '''
    # XXX: BUG WAS HERE
    material_order = [ic.AL, ic.TE, ic.BE, ic.CEBR3]
    materials = []
    for mat_name in material_order:
        if mat_name == ic.AL:
            # to be modified
            thick = np.nan
        else:
            thick = ic.THICKNESSES[mat_name]    # cm
        rho = ic.DENSITIES[mat_name]            # g / cm3
        atten_dat = AttenuationData.from_nist_file(ic.ATTEN_FILES[mat_name])
        materials.append(Material(ic.DIAMETER, thick, rho, atten_dat))
    return materials


def relevant_to_sims(ds, dim, fspec, t):
    area = np.ones(dim) * ic.FULL_AREA / 4
    ds.materials[0].thickness = t
    # energy resolution "smears" the result. we don't want that, yet.
    unsmeared = ds.generate_detector_response_to(fspec, disperse_energy=False)
    eff_area = np.matmul(unsmeared, area)
    smeared = ds.apply_detector_dispersion_for(fspec, unsmeared)
    # save effective area and response matrix to a single compressed file
    save_fname = f"{fspec.goes_class}_{t:.2e}_resp-and-area"
    save_fname = os.path.join(ic.DATA_DIR, save_fname)
    return unsmeared, eff_area, smeared, save_fname


def saveit(fspec, unsmeared, eff_area, smeared, save_fname, al_thickness):
    np.savez_compressed(
            save_fname, flare_spectrum=fspec.flare,
            energies=fspec.energies, response_matrix=smeared, eff_area=eff_area)
    print(f"saved {fspec.goes_class} at Al = {al_thickness:.2e} cm thick")


def sim_with_thicks(flare_ident, det_stack, fspec, al_thicks):
    if not os.path.exists(ic.DATA_DIR): os.mkdir(ic.DATA_DIR)
    dimension = fspec.energies.size
    # unmodified area vector (i.e. at each energy, the detector has the same area. ideal case.)
    for t in al_thicks:
        unsmeared, eff_area, smeared, save_fname = relevant_to_sims(det_stack, dimension, fspec, t)
        saveit(fspec, unsmeared, eff_area, smeared, save_fname, t)


def diagnostic(dstack, flare_spectrum, goes_name, remaining):
    al_thick_bounds = 1e-4 * np.array([10, 200])     # cm
    dt = 1e-3                                        # cm
    al_thicknesses = np.arange(*al_thick_bounds, step=dt)
    print(f"{remaining * al_thicknesses.size} files remaining")
    sim_with_thicks(goes_name, dstack, flare_spectrum, al_thicknesses)


def calc_target_cps(dead_time):
    # target counts for 95% confidence that there is no pileup
    # credit: Trevor's dissertation (wherever he got it from! :> )
    return -1 * np.log(0.95) / dead_time    # counts / second


def count_edge(cts, target, dt):
    '''
    when we want to stop.
        with positive increment, cts < target, i.e. too much attenuation
        with negative increment, cts > target, i.e. attenuator is too thin
    '''
    if dt > 0: return cts < target
    if dt < 0: return cts > target
    else: raise ValueError("dt indistinguishable from zero")


def find_appropriate_counts(detector_stack, flare_spectrum, dead_time, start_thick):
    '''
    start with thickness that's sure to attenuate the flare
    zigzag around target count rate until we're close enough for gov't work (i.e. within 1e-9 cm Al thicknessof it)
    '''
    E_MIN, E_MAX = 8, 100                         # keV
    fl, en = flare_spectrum.flare, flare_spectrum.energies
    cur_thick = start_thick                 # cm
    step = -1 * start_thick / 10            # cm
    target_cps = calc_target_cps(dead_time) # counts / sec

    TARGET_DIVS = 8
    TOL = 0.05
    divs = 0
    # make step size smaller max. TARGET_DIVS times
    while divs < TARGET_DIVS:
        print(f"{flare_spectrum.goes_class}: {cur_thick:.3e} cm")
        unsmeared, eff_area, smeared, save_fname = relevant_to_sims(detector_stack, en.size, flare_spectrum, cur_thick)
        counts_per_kev = np.matmul(smeared, fl) * eff_area
        restrict = np.logical_and(en > E_MIN, en < E_MAX)
        cur_counts = np.trapz(counts_per_kev[restrict], x=en[restrict])
        if count_edge(cur_counts, target_cps, step):
            print("Found the count edge.\n", f"Counts: {cur_counts}, thickness: {cur_thick:.3e} cm")
            step /= -10
            divs += 1

        cur_thick += step
        delta = 1 - cur_counts/target_cps
        if abs(delta) < TOL and delta > 0:
            break

    if divs == TARGET_DIVS:
        print("** Hit max number of step divisions.")
    # go back one
    cur_thick -= step
    saveit(flare_spectrum, unsmeared, eff_area, smeared, save_fname, cur_thick)


def main():
    # if not os.path.exists(ic.LOGS_DIR):
    #     os.mkdir(ic.LOGS_DIR)
    # log = logging.getLogger()
    e_start = 1.0   # keV
    e_end = 299.0   # keV
    de = 0.1        # keV
    goes_classes = ['C1', 'C5', 'M1', 'M5', 'X1']
    bs = gen_base_stack()
    # thickness to be updated
    ds = HafxStack(bs, Sipm3000())
    for i, f in enumerate(goes_classes):
        print(f"Starting {f}")
        flare_spectrum = FlareSpectrum.make_with_battaglia_scaling(f, e_start, e_end, de)
        dead_time = 1e-6    # s
        start_thick = 0.1   # cm (way too thick!)
        find_appropriate_counts(ds, flare_spectrum, dead_time, start_thick)
        # diagnostic(ds, flare_spectrum, f, len(goes_classes) - i)


if __name__ == '__main__': main()
