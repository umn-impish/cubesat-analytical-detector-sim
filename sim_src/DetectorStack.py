import numpy as np
from .AttenuationData import AttenuationType
from .FlareSpectrum import FlareSpectrum
from .PhotonDetector import PhotonDetector

class DetectorStack:
    '''
    Represents the "stack" of materials on top of a detector as well as the photon detector itself.
    '''
    # XXX: we are dragging around a FlareSpectrum object because eventually Compton scattering may need
    # access to the energy vector (because it is energy nonconserving) and detector energy dispersion
    # also needs the energy vector.
    def __init__(self, materials: list, photon_detector: PhotonDetector):
        '''position zero in the materials array corresponds to the outermost one'''
        self.materials = materials
        self.photon_detector = photon_detector

    def generate_detector_response_to(
            self, incident_spectrum: FlareSpectrum, disperse_energy: bool, chosen_attenuations: list=AttenuationType.ALL) -> np.ndarray:
        response = self._generate_attenuation_response_due_to(incident_spectrum, chosen_attenuations)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

    def apply_detector_dispersion_for(self, incident_spectrum: FlareSpectrum, resp_matrix: np.ndarray) -> np.ndarray:
        pd_spread = self.photon_detector.generate_energy_resolution_given(incident_spectrum)
        return np.matmul(pd_spread, resp_matrix)

    def _generate_material_response_due_to(self, incident_spectrum: FlareSpectrum, attenuations: list) -> np.ndarray:
        ''' don't call this directly. doesn't include scintillator effects. '''
        response = np.identity(incident_spectrum.energies.size)
        for material in self.materials:
            response = np.matmul(material.generate_overall_response_matrix_given(incident_spectrum, attenuations), response)
        return response

    def _dispatch_dispersion(self, incident_spectrum: FlareSpectrum, response: np.ndarray, do_it: bool):
        if do_it: return self.apply_detector_dispersion_for(incident_spectrum, response)
        else: return response

    @property
    def area(self):
        return self.materials[0].area


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
            # we must include photoelectric absorption. it is how the scintillator works.
            abs_atts = list(set([AttenuationType.PHOTOELECTRIC_ABSORPTION] + chosen_attenuations))
            absorbed = ident - self.scintillator.generate_overall_response_matrix_given(incident_spectrum, abs_atts)
            response = np.matmul(absorbed, response)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

