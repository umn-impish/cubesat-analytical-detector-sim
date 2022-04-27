import numpy as np
from ..sim_src.PhotonDetector import PhotonDetector
from ..sim_src.FlareSpectrum import FlareSpectrum

class LinearInterpolateDisperseDetector(PhotonDetector):
    def __init__(self, e1, e2, fwhm1, fwhm2):
        self.slope = (fwhm2 - fwhm1) / (e2 - e1)
        self.intercept = fwhm1 - self.slope*e1
        super().__init__()

    def interpolate_energy_resolution(self, energies: np.ndarray) -> np.ndarray:
        ''' put a line through two points that we know (667 from Epic Crystal, 20 keV from a meeting with John) '''
        return (self.slope*energies + self.intercept)

    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        step_size = incident_spectrum.energies[1] - incident_spectrum.energies[0]
        resolutions = self.interpolate_energy_resolution(incident_spectrum.energies)
        # convert the energy fwhm to "index" space
        fwhm = resolutions * incident_spectrum.energies / step_size

        dim = incident_spectrum.energies.size
        rng = np.arange(dim)
        vectorized_indices = np.tile(rng, (dim, 1)).transpose()
        return gaussian_row(dim, fwhm, vectorized_indices)


def gaussian_row(dim, fwhm, idx):
    sd = fwhm / (2 * np.sqrt(2 * np.log(2)))
    space = np.arange(0, dim, dtype=np.float64)
    prefac = 1 / (sd * np.sqrt(2 * np.pi))
    exponent = -(space - idx)**2 / (2 * sd*sd)
    return  prefac * np.exp(exponent)
