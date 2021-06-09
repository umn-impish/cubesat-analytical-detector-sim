import numpy as np
from FlareSpectrum import FlareSpectrum

class PhotonDetector:
    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        raise NotImplementedError

class Sipm3000(PhotonDetector):
    # guess for now
    DELTA_E = 0.4

    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        # fwhm to standard deviation
        sd = 2 * np.sqrt(2 * np.log(2)) * Sipm3000.DELTA_E
        # number of rows
        dim = incident_spectrum.energies.shape[0]
        res_mat = np.zeros((dim, dim))
        rng = np.arange(dim)
        for i in rng:
            res_mat[i] = gaussian_row(dim, sd, i)
        return res_mat

def gaussian_row(dim, sd, idx):
    # standard deviation is in percentage
    sd = dim * sd
    space = np.arange(0, dim, dtype=np.float64)
    prefac = 1
    exponent = -(space - idx)**2 / (2 * sd*sd)
    return  prefac * np.exp(exponent)
