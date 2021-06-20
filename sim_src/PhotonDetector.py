import numpy as np
from .FlareSpectrum import FlareSpectrum

class PhotonDetector:
    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        raise NotImplementedError("Subclass PhotonDetector to implement detector-specific behavior.")

class Sipm3000(PhotonDetector):
    # guess for now
    DELTA_E = 0.4

    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        step_size = incident_spectrum.energies[1] - incident_spectrum.energies[0]
        fwhm = self.DELTA_E * incident_spectrum.energies / step_size
        sd = fwhm / (2 * np.sqrt(2 * np.log(2)))

        dim = incident_spectrum.energies.size
        rng = np.arange(dim)
        # vectorized NumPy operations are multi-threaded and run in C;
        # generally much faster than Python loops
        vectorized_indices = np.tile(rng, (dim, 1)).transpose()
        return gaussian_row(dim, sd, vectorized_indices)


def gaussian_row(dim, sd, idx):
    space = np.arange(0, dim, dtype=np.float64)
    prefac = 1 / (sd * np.sqrt(2 * np.pi))
    exponent = -(space - idx)**2 / (2 * sd*sd)
    return  prefac * np.exp(exponent)
