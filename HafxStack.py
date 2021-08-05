import numpy as np

from sim_src.AttenuationData import AttenuationType
from sim_src.DetectorStack import DetectorStack
from sim_src.FlareSpectrum import FlareSpectrum
from sim_src.PhotonDetector import PhotonDetector


class HafxStack(DetectorStack):
    ''' photoabsorption into the scintillator crystal is different here so we need separate behavior. '''
    def __init__(self, materials: list, photon_detector: PhotonDetector, enable_scintillator: bool=True):
        super().__init__(materials, photon_detector)
        # take off the scintillator to treat it separately
        self.scintillator = self.materials.pop()
        # XXX: set to True to disable the scintillator! (either during construction or afterwards)
        self.enable_scintillator = enable_scintillator

    def generate_detector_response_to(
            self, incident_spectrum: FlareSpectrum, disperse_energy: bool, chosen_attenuations: list=AttenuationType.ALL) -> np.ndarray:
        response = self._generate_material_response_due_to(incident_spectrum, chosen_attenuations)
        if self.enable_scintillator:
            # now incorporate the scintillator
            ident = np.identity(incident_spectrum.energies.size)
            # we must include photoelectric absorption as this mechanism leads to scintillation.
            abs_atts = list(set([AttenuationType.PHOTOELECTRIC_ABSORPTION] + chosen_attenuations))
            absorbed = ident - self.scintillator.generate_overall_response_matrix_given(incident_spectrum, abs_atts)
            response = np.matmul(absorbed, response)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

