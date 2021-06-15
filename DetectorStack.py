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

    def generate_detector_response_to(self, incident_spectrum: FlareSpectrum, disperse_energy: bool) -> np.ndarray:
        response = self.generate_attenuation_response_due_to(incident_spectrum)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

    def generate_attenuation_response_due_to(self, incident_spectrum) -> np.ndarray:
        response = np.identity(incident_spectrum.energies.shape[0])
        # BUG WAS HERE :(
        for material in self.materials:
            response = np.matmul(material.generate_overall_response_matrix_given(incident_spectrum), response)
        return response

    def apply_detector_dispersion_for(self, incident_spectrum: FlareSpectrum, resp_matrix: np.ndarray) -> np.ndarray:
        pd_spread = self.photon_detector.generate_energy_resolution_given(incident_spectrum)
        return np.matmul(pd_spread, resp_matrix)

    def _dispatch_dispersion(self, incident_spectrum: FlareSpectrum, response: np.ndarray, do_it: bool):
        if do_it: return self.apply_detector_dispersion_for(incident_spectrum, response)
        else: return response

    def generate_effective_area_due_to(self, incident_spectrum) -> np.ndarray:
        # area vector
        av = np.ones(incident_spectrum.energies.size) * self.area
        return np.matmul(self.generate_detector_response_to(incident_spectrum, disperse_energy=False), av)

    def respond_to(self, incident_spectrum) -> np.ndarray:
        self.generate_detector_response_to(incident_spectrum) * incident_spectrum.energies

    @property
    def area(self):
        return self.materials[0].area


class HafxStack(DetectorStack):
    ''' photoabsorption into the scintillator crystal is different here so we need separate behavior. '''
    def __init__(self, materials: list, photon_detector: PhotonDetector):
        super().__init__(materials, photon_detector)
        # take off the scintillator to treat it separately
        self.scintillator = self.materials.pop()

    def generate_detector_response_to(
            self, incident_spectrum: FlareSpectrum, disperse_energy: bool=True) -> np.ndarray:
        response = self.generate_attenuation_response_due_to(incident_spectrum)
        # now incorporate the scintillator
        ident = np.identity(incident_spectrum.energies.shape[0])
        absorbed = ident - self.scintillator.generate_overall_response_matrix_given(incident_spectrum)
        response = np.matmul(absorbed, response)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

