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
        response = self.generate_attenuation_response_due_to(incident_spectrum)
        pd_spread = self.photon_detector.generate_energy_resolution_given(incident_spectrum)
        response = np.matmul(pd_spread, response)
        return response

    def generate_attenuation_response_due_to(self, incident_spectrum) -> np.ndarray:
        response = np.identity(incident_spectrum.energies.shape[0])
        for material in self.materials[:-1]:
            response = np.matmul(material.generate_overall_response_matrix_given(incident_spectrum), response)
        return response

    def generate_effective_area_due_to(self, incident_spectrum) -> np.ndarray:
        self.generate_detector_response_to(incident_spectrum) * self.materials[0].area

    def respond_to(self, incident_spectrum) -> np.ndarray:
        self.generate_detector_response_to(incident_spectrum) * incident_spectrum.energies

    @property
    def area(self):
        return self.materials[0].area


class HafxStack(DetectorStack):
    ''' photoabsorption into the scintillator crystal is different here so we need separate behavior. '''
    def __init__(self, materials: list, photon_detector: PhotonDetector):
        super().__init__(materials, photon_detector)
        # take off the scintillator to treat it separately.
        self.scintillator = self.materials.pop()

    def generate_detector_response_to(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        response = self.generate_attenuation_response_due_to(incident_spectrum)
        # now incorporate the scintillator
        ident = np.identity(incident_spectrum.energies.shape[0])
        absorbed = ident - self.scintillator.generate_overall_response_matrix_given(incident_spectrum)
#        for x in absorbed:
#            print(absorbed)
#        input()
        response = np.matmul(absorbed, response)
        pd_spread = self.photon_detector.generate_energy_resolution_given(incident_spectrum)
        response = np.matmul(pd_spread, response)
        return response
