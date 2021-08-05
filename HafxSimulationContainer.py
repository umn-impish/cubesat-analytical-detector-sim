import numpy as np
import os
import sys

import sim_src.impress_constants as ic
from sim_src.FlareSpectrum import FlareSpectrum
from sim_src.AttenuationData import AttenuationData
from sim_src.Material import Material
from sim_src.PhotonDetector import Sipm3000
from HafxStack import HafxStack

def gen_hafx_stack(al_thick: np.float64):
    ''' put the HaFX materials in the right order (variable aluminum thickness)'''
    mat_order = ic.HAFX_MATERIAL_ORDER
    materials = []
    for name in mat_order:
        thick = al_thick if name == ic.AL else ic.THICKNESSES[name]
        rho = ic.DENSITIES[name]
        atten_dat = AttenuationData.from_nist_file(ic.ATTEN_FILES[name])
        materials.append(Material(ic.DIAMETER, thick, rho, atten_dat))
    return HafxStack(materials, Sipm3000())


class HafxSimulationContainer:
    MIN_THRESHOLD_ENG = ic.MIN_THRESHOLD_ENG
    MAX_THRESHOLD_ENG = ic.MAX_THRESHOLD_ENG

    KAL_THICKNESS = 'al_thickness'
    KFLARE_THERMAL = 'thermal'
    KFLARE_NONTHERMAL = 'nonthermal'
    KENERGIES = 'energies'
    KPURE_RESPONSE = 'pure_response_matrix'
    KDISPERSED_RESPONSE = 'dispersed_response_matrix'
    KEFFECTIVE_AREA = 'effective_area'
    KGOES_CLASS = 'goes_class'
    MATRIX_KEYS = (
        KDISPERSED_RESPONSE, KPURE_RESPONSE
    )
    DEFAULT_SAVE_DIR = 'responses-and-areas'

    @classmethod
    def from_saved_file(cls, filename: str):
        ''' load the container from a (compressed) .npz file '''
        data = np.load(filename)

        try:
            goes_class = data[cls.KGOES_CLASS]
        except KeyError as e:
            goes_class = filename.split('_')[-3]
        fs = FlareSpectrum(
                goes_class,
                data[cls.KENERGIES],
                data[cls.KFLARE_THERMAL],
                data[cls.KFLARE_NONTHERMAL])

        ret = cls(aluminum_thickness=data[cls.KAL_THICKNESS], flare_spectrum=fs)
        ret.flare_spectrum = fs
        for k in cls.MATRIX_KEYS:
            ret.matrices[k] = data[k]
        return ret

    def __init__(self, aluminum_thickness: np.float64=None, flare_spectrum: FlareSpectrum=None):
        self.detector_stack = gen_hafx_stack(aluminum_thickness)
        self.al_thick = aluminum_thickness
        self.matrices = {k: None for k in self.MATRIX_KEYS}
        self.flare_spectrum = flare_spectrum

    @property
    def al_thick(self):
        return self.__al_thick

    @al_thick.setter
    def al_thick(self, new):
        self.__al_thick = new
        self.detector_stack.materials[0].thickness = new

    def compute_effective_area(self, cps_threshold: np.int64=0):
        if cps_threshold > 0:
            restrict = np.logical_and(
                    self.flare_spectrum.energies >= self.MIN_THRESHOLD_ENG,
                    self.flare_spectrum.energies <= self.MAX_THRESHOLD_ENG)
            dispersed_flare = np.matmul(self.matrices[self.KDISPERSED_RESPONSE], self.flare_spectrum.flare)
            relevant_cps = np.trapz(
                dispersed_flare[restrict] * ic.SINGLE_DET_AREA, x=self.flare_spectrum.energies[restrict])

            # "set" effective area to zero if we get more than the threshold counts
            if relevant_cps > cps_threshold:
                return np.zeros_like(self.flare_spectrum.energies)

        area_vector = np.ones_like(self.flare_spectrum.energies) * ic.SINGLE_DET_AREA
        att_area = np.matmul(self.matrices[self.KPURE_RESPONSE], area_vector)
        return att_area

    def simulate(self):
        if self.al_thick is None:
            raise ValueError("Aluminum thickness has not been set.")
        # get the un-dispersed (pure) response matrix
        self.matrices[self.KPURE_RESPONSE] =\
                self \
                .detector_stack \
                .generate_detector_response_to(self.flare_spectrum, False) \
        # apply CeBr3 energy resolution
        self.matrices[self.KDISPERSED_RESPONSE] =\
                self \
                .detector_stack \
                .apply_detector_dispersion_for(self.flare_spectrum, self.matrices[self.KPURE_RESPONSE])

    def gen_file_name(self, prefix):
        return f"{prefix}_{self.flare_spectrum.goes_class}_{self.al_thick:.3e}cm_hafx"

    def save_to_file(self, out_dir=DEFAULT_SAVE_DIR, prefix=None):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        ''' save object data into a file that can be loaded back in later '''
        if None in self.matrices:
            raise ValueError("Matrices haven't been computed so we can't save them.")
        if self.flare_spectrum is None:
            raise ValueError("Stored FlareSpectrum is None--simulation probably hasn't run.")

        to_save = dict()
        to_save[self.KAL_THICKNESS] = self.al_thick
        to_save[self.KFLARE_THERMAL] = self.flare_spectrum.thermal
        to_save[self.KFLARE_NONTHERMAL] = self.flare_spectrum.nonthermal
        to_save[self.KENERGIES] = self.flare_spectrum.energies

        for k in self.MATRIX_KEYS:
            to_save[k] = self.matrices[k]
        outfn = os.path.join(out_dir, self.gen_file_name(prefix or ''))
        np.savez_compressed(outfn, **to_save)

