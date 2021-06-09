import numpy as np
from AttenuationData import AttenuationData, AttenuationType
from FlareSpectrum import FlareSpectrum

class Material:
    def __init__(self, diameter: np.float64, attenuation_thickness: np.float64,
                 mass_density: np.float64, attenuation_data: AttenuationData):
        self.diameter = diameter
        self.mass_density = mass_density
        self.attenuation_data = attenuation_data
        self.thickness = attenuation_thickness

    def generate_overall_response_matrix_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        dim = incident_spectrum.shape[0]
        mat = np.identity(dim, dtype=np.float64)
        # everything is diagonal except Compton scattering, so the matrices commute
        # i.e. multiplication order doesn't matter
        for k in self.attenuation_data.keys():
            mat *= self.generate_modifying_matrix_for(k, incident_spectrum)

        return mat

    def generate_modifying_matrix_for(self, data_type: int, incident_spectrum: FlareSpectrum) -> np.ndarray:
        modify_gen_lookup = {
            AttenuationType.PHOTOELECTRIC_ABSORPTION: self._gen_phot_ray,
            AttenuationType.RAYLEIGH: self._gen_phot_ray,
            AttenuationType.COMPTON: self._gen_compton,
        }
        return modify_gen_lookup(incident_spectrum)

    def _gen_phot_ray(self, which: int, incident_spectrum: FlareSpectrum) -> np.ndarray:
        interpolated_attenuation = self.attenuation_data.interpolate_from(incident_spectrum)
        att_dat = interpolated_attenuation[which]
        exponent = -1 * att_dat * self.mass_density * self.thickness
        attenuate = np.exp(exponent)

        # return attenuation as a matrix rather than a vector
        return np.diag(attenuate)

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
