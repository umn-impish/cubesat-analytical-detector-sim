import numpy as np
from scipy import interpolate
from .FlareSpectrum import FlareSpectrum

class AttenuationType:
    PHOTOELECTRIC_ABSORPTION = 1
    RAYLEIGH = 2
    COMPTON = 3
    ALL = [PHOTOELECTRIC_ABSORPTION, RAYLEIGH, COMPTON]

    @staticmethod
    def named():
        return {
            k: n for (k, n) in zip(
                AttenuationType.ALL,
                ('photoelectric', 'rayleigh', 'compton')
            )
        }

class AttenuationData:
    @classmethod
    def from_nist_file(cls, file_path: str):
        '''
        assumes:
            - tab-delimited
            - energy first col
            - photoelectric attenuation second col
            - rayleigh third col
            - compton fourth col
        '''
        # "unpack" transposes the file contents
        energies, photo, rayleigh, compton = np.loadtxt(file_path, dtype=np.float64, unpack=True)
        return cls(energies, photo, rayleigh, compton)

    def __init__(self, energies: np.ndarray, photoelec: np.ndarray, rayleigh: np.ndarray, compton: np.ndarray):
        '''call this or the from_file class method to construct the object'''
        self.energies = energies
        self.attenuations = dict()
        self.attenuations[AttenuationType.PHOTOELECTRIC_ABSORPTION] = photoelec
        self.attenuations[AttenuationType.RAYLEIGH] = rayleigh
        self.attenuations[AttenuationType.COMPTON] = compton

    def interpolate_from(self, incident_flare: FlareSpectrum) -> dict:
        ''' interpolate standard data to fit given incident_flare energy spectrum
            returns a function that interpolates log of attenuation
            data (to be integrated or evaluated)
        '''
        interp_log_att_funcs = dict()
        for key, att in self.attenuations.items():
            # interpolate between NIST energies
            # we want straight-line interpolation on the log plot,
            # so take the log before doing any fitting
            loge, logat = np.log(self.energies), np.log(att)
            interp_func = interpolate.interp1d(
                x=loge, y=logat,
                fill_value="extrapolate"
            )
            interp_log_att_funcs[key] = interp_func
        return interp_log_att_funcs
