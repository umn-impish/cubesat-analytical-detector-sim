import os
import numpy as np

import impress_constants as ic
from AttenuationData import AttenuationData
from DetectorStack import HafxStack, GoesHashedHafxStack
from FlareSpectrum import FlareSpectrum, goes_class_lookup
from Material import Material, GoesHashedMaterial
from PhotonDetector import Sipm3000

DATA_DIR = 'responses-and-areas'

def gen_base_stack():
    ''' make the materials in every detector stack '''
    material_order = [ic.TE, ic.BE, ic.CEBR3]
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


def sim_with_thicks(flare_ident, det_stack, fspec, al_thicks):
    dimension = fspec.energies.size
    # unmodified area vector (i.e. at each energy, the detector has the same area. ideal case.)
    area = np.ones(dimension) * ic.FULL_AREA / 4
    for t in al_thicks:
        det_stack.materials[0].thickness = t
        # energy resolution "smears" the result. we don't want that, yet.
        unsmeared = det_stack.generate_detector_response_to(fspec, disperse_energy=False)
        eff_area = np.matmul(unsmeared, area)
        smeared = det_stack.apply_detector_dispersion_for(fspec, unsmeared)
        # save effective area and response matrix to a single compressed file
        save_fname = f"{flare_ident}_{t:.2e}_resp_and_area"
        save_fname = os.path.join(DATA_DIR, save_fname)
        np.savez_compressed(
                save_fname, flare_spectrum=fspec.flare,
                energies=fspec.energies, resp=smeared, eff_area=eff_area)
        print(f"done with {flare_ident} at Al = {t:.2e} cm thick")


def main():
    base_stack = gen_base_stack()
    # thickness to be updated
    dstack = HafxStack(base_stack, Sipm3000())

    flares = ['C1', 'C5', 'M1', 'M5', 'X1']
    al_thick_bounds = [x * 1e-4 for x in (10, 200)]  # cm
    dt = 1e-3                                        # cm
    al_thicknesses = np.arange(*al_thick_bounds, step=dt)
    print("Number of files to generate:", len(flares) * al_thicknesses.size)
    input("Continue? (ctrl c to quit)")

    e_start = 1.0   # keV
    e_end = 150.0   # keV
    de = 0.05       # keV

    for f in flares:
        g = goes_class_lookup(f)
        flare_spectrum = FlareSpectrum.make_with_battaglia_scaling(g, e_start, e_end, de)
        sim_with_thicks(f, dstack, flare_spectrum, al_thicknesses)

if __name__ == '__main__': main()
