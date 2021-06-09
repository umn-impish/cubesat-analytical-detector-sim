import numpy as np
import os
import sys

from impress_constants import *
from Material import Material
from DetectorStack import DetectorStack
from Instrument import Instrument

def stack_with_thickness(al_thick):
    material_order = [AL, TE, BE, CEBR3]
    materials = []
    for mat_name in material_order:
        # use passed-in thickness if it's aluminum
        thick = THICKNESSES[mat_name] if mat_name != AL else al_thick   #cm
        rho = DENSITIES[mat_name]                                       # g / cm3
        atten_dat = AttenuationData.from_nist_file(ATTEN_FILES[mat_name])
        materials.append(Material(diameter, thick, rho, atten_dat))
    return materials

def build_impress(aluminum_thicknesses):
    # order is: Al, Teflon, Be, CeBr3
    detector_stacks = []
    for al_thick in aluminum_thicknesses:
        detector_stacks.append(stack_with_thickness(al_thick))
    return Instrument(detector_stacks)

def gen_flare_spectrum(goes_flux):
    pass
    # TODO:
    # implement Battaglia scaling class here
    # get output from f_vth from Battaglia temperature
    # get output from f_1pow for Battaglia 35 keV flux
    # sum them and return

def main():
    goes_flux = 1e-5                        # W / cm2
    al_thick = [10, 25, 120, 190] * 1e-4    # cm
    e_start = 1                             # keV
    e_end = 300                             # keV

    impress = build_impress(al_thick)
    flare_spectrum = gen_flare_spectrum(e_start, e_end, goes_flux)
    responses = []
    for det in impress.detector_stacks:
        responses.append(det.generate_detector_response_to(flare))
    # ...

if __name__ == '__main__': main()
