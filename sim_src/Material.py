import numpy as np
from .AttenuationData import AttenuationData, AttenuationType
from .FlareSpectrum import FlareSpectrum

class Material:
    def __init__(self, diameter: np.float64, attenuation_thickness: np.float64,
                 mass_density: np.float64, attenuation_data: AttenuationData):
        self.diameter = diameter
        self.mass_density = mass_density
        self.attenuation_data = attenuation_data
        self.thickness = attenuation_thickness

    @property
    def area(self):
        return (self.diameter / 2)**2 * np.pi

    def generate_overall_response_matrix_given(self, incident_spectrum: FlareSpectrum, attenuations: list) -> np.ndarray:
        dim = incident_spectrum.energies.shape[0]
        mat = np.identity(dim, dtype=np.float64)
        # everything is diagonal except Compton scattering, so the matrices commute
        # i.e. multiplication order doesn't matter
        for k in attenuations:
            mat = np.matmul(self.generate_modifying_matrix_for(k, incident_spectrum), mat)
        return mat

    def generate_modifying_matrix_for(self, mechanism_type, incident_spectrum: FlareSpectrum) -> np.ndarray:
        modify_gen_lookup = {
            AttenuationType.PHOTOELECTRIC_ABSORPTION: self._gen_photo,
            AttenuationType.RAYLEIGH: self._gen_rayleigh,
            AttenuationType.COMPTON: self._gen_compton,
        }
        return modify_gen_lookup[mechanism_type](incident_spectrum)

    def _gen_phot_ray(self, which, incident_spectrum: FlareSpectrum) -> np.ndarray:
        interpolated_attenuations = self.attenuation_data.interpolate_from(incident_spectrum)
        relevant_attenuation = interpolated_attenuations.attenuations[which]
        exponent = -1 * relevant_attenuation * self.mass_density * self.thickness
        actual_attenuation = np.exp(exponent)
        return np.diag(actual_attenuation)

    def _gen_photo(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        return self._gen_phot_ray(AttenuationType.PHOTOELECTRIC_ABSORPTION, incident_spectrum)

    def _gen_rayleigh(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        return self._gen_phot_ray(AttenuationType.RAYLEIGH, incident_spectrum)

    def _gen_compton(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        '''
        NB: not implemented!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        # do nothing
        return np.identity(incident_spectrum.energies.shape[0])
