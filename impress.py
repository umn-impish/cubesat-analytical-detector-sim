import numpy as np

import impress_constants as ic
from AttenuationData import AttenuationData
from DetectorStack import HafxStack
from FlareSpectrum import FlareSpectrum, goes_class_lookup
from Instrument import Instrument
from Material import Material
from PhotonDetector import Sipm3000

def base_stack():
    material_order = [ic.TE, ic.BE, ic.CEBR3]
    materials = []
    for mat_name in material_order:
        thick = ic.THICKNESSES[mat_name]                                      # cm
        rho = ic.DENSITIES[mat_name]                                          # g / cm3
        atten_dat = AttenuationData.from_nist_file(ic.ATTEN_FILES[mat_name])
        materials.append(Material(ic.DIAMETER, thick, rho, atten_dat))

    return materials


def build_impress(aluminum_thicknesses):
    # order is: Al, Teflon, Be, CeBr3
    detector_stacks = []
    base_mat_stack = base_stack()
    for al_thick in aluminum_thicknesses:
        ad = AttenuationData.from_nist_file(ic.ATTEN_FILES[ic.AL])
        alum = Material(
                ic.DIAMETER, al_thick, ic.DENSITIES[ic.AL], ad)
        # all DetectorStacks reference the same base material objects
        # therefore, their saved matrix hashes are shared.
        cur_stack = [alum] + base_mat_stack
        detector_stacks.append(HafxStack(cur_stack, Sipm3000()))
    return Instrument(detector_stacks)


def impress_flare_spectrum(goes_flux, e_start, e_end, de):
    # save a hashmap of flares so that it doesn't recompute every time (slow)
    try:
        return impress_flare_spectrum.hash[goes_flux]
    except (KeyError, AttributeError) as e:
        spec = FlareSpectrum.make_with_battaglia_scaling(goes_flux, e_start, e_end, de)
        if isinstance(e, KeyError):
            impress_flare_spectrum.hash[goes_flux] = spec
        else:
            impress_flare_spectrum.hash = { goes_flux : spec }
        return spec


# TODO: add a goes_flux variable to FlareSpectrum to use for hashing elsewhere
def test():
    goes_class = 'M1'
    goes_flux = goes_class_lookup(goes_class)       # W / cm2
    al_thicks = np.array([10, 20, 120, 190]) * 1e-4 # cm
    e_start = 1                                     # keV
    e_end = 100                                     # keV
    de = 0.05                                       # keV

    impress = build_impress(al_thicks)
    flare_spectrum = impress_flare_spectrum(goes_flux, e_start, e_end, de)
    count_vectors = []
    # eff_area_vecs = []
    for det in impress.detector_stacks:
        response = det.generate_detector_response_to(flare_spectrum)
        count_vectors.append(np.matmul(response, flare_spectrum.flare))
        # base_area = np.ones(flare_spectrum.energies.size) * det.area
        # eff_area_vecs.append(np.matmul(response, base_area))

    outf = 'flare_out.tab'
    out = [flare_spectrum.energies] + [e for e in count_vectors] + [flare_spectrum.flare]
    # out = [flare_spectrum.energies] + [a for a in eff_area_vecs]
    np.savetxt(outf, np.transpose(out))
#     total = np.trapz(count_vectors, x=flare_spectrum.energies, dx=de, axis=1) * impress.detector_stacks[0].area
#     print(f"Total counts: {total}")

if __name__ == '__main__': test()
