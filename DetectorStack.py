import copy
import numpy as np
from FlareSpectrum import FlareSpectrum
from PhotonDetector import PhotonDetector

class DetectorStack:
    '''
    Represents the "stack" of materials on top of a detector as well as the photon detector itself.
    '''
    def __init__(self, materials: list, photon_detector: PhotonDetector):
        '''position zero in the materials array corresponds to the outermost one'''
        self.materials = materials
        self.photon_detector = photon_detector

    def generate_detector_response_to(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        # make a copy so we don't accidenetally modify the initial values 
        spectrum = copy.deepcopy(incident_spectrum)
        # start with identity matrix
        response = np.diag(spectrum.energies.shape[0])
        # attenuate due to all materials
        for mat in self.materials:
            response *= mat.generate_overall_response_matrix_given(spectrum)
        # smear energies out due to photon detector finite energy resolution
        pd_spread = self.photon_detector.generate_energy_resolution_given(spectrum)
        response *= pd_spread

        return response

    def generate_effective_area_due_to(self, incident_spectrum) -> np.ndarray:
        self.generate_detector_response_to(incident_spectrum) * self.materials[0].area

    def respond_to(self, incident_spectrum) -> np.ndarray:
        self.generate_detector_response_to(incident_spectrum) * incident_spectrum.energies
