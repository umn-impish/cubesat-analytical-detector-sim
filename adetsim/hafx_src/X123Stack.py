import numpy as np

from ..sim_src.AttenuationData import AttenuationType, AttenuationData
from ..sim_src.DetectorStack import DetectorStack
from ..sim_src.FlareSpectrum import FlareSpectrum
from ..sim_src.Material import Material

from . import HafxMaterialProperties as hmp
from .X123ResolutionDetector import X123ResolutionDetector


class X123Stack(DetectorStack):
    ''' material stack for X-123 detector from AmpTek '''
    X123_DIAMETER = 2.82 * 2 / 10          # cm
    DET_THICK = 500e-6 * hmp.METER_PER_CM  # cm
    BE_THICK = 8e-6 * hmp.METER_PER_CM     # cm
    def __init__(self, detector_material=hmp.SI):
        materials = [
            Material(
                diameter=self.X123_DIAMETER,
                attenuation_thickness=self.BE_THICK,
                mass_density=hmp.DENSITIES[hmp.BE],
                attenuation_data=AttenuationData.from_nist_file(hmp.ATTEN_FILES[hmp.BE]))]
        self.detector_volume = Material(
            diameter=self.X123_DIAMETER,
            attenuation_thickness=self.DET_THICK,
            mass_density=hmp.DENSITIES[hmp.SI],
            attenuation_data=AttenuationData.from_nist_file(hmp.ATTEN_FILES[detector_material]))
        super().__init__(materials, X123ResolutionDetector())

    def generate_detector_response_to(
            self, spectrum: FlareSpectrum, disperse_energy: bool,
            chosen_attenuations: list=AttenuationType.ALL) -> np.ndarray:
        prelim_resp = self._generate_material_response_due_to(spectrum, chosen_attenuations)
        abzorbed = np.eye(spectrum.energies.size) -\
            self.detector_volume.generate_overall_response_matrix_given(
                spectrum, [AttenuationType.PHOTOELECTRIC_ABSORPTION])
        return self._dispatch_dispersion(spectrum, abzorbed @ prelim_resp, disperse_energy)
