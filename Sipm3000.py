import numpy as np
from sim_src.PhotonDetector import PhotonDetector
from sim_src.FlareSpectrum import FlareSpectrum

# XXX: the photon detector isn't the main constraint on energy resolution. the scintillator does.
#      => move photon detector stuff to scintillator.
class Sipm3000(PhotonDetector):
    # guess for now
    DE_20KEV = 0.4
    DE_667KEV = 0.045
    SLOPE = (DE_667KEV - DE_20KEV) / (667 - 20)
    INTERCEPT = DE_20KEV - SLOPE*20

    def interpolate_energy_resolution(self, energies: np.ndarray) -> np.ndarray:
        ''' put a line through two points that we know (667 from Epic Crystal, 20 keV from a meeting with John) '''
        return (self.SLOPE*energies + self.INTERCEPT)

    def generate_energy_resolution_given(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        step_size = incident_spectrum.energies[1] - incident_spectrum.energies[0]
        resolutions = self.interpolate_energy_resolution(incident_spectrum.energies)
        fwhm =  resolutions * incident_spectrum.energies / step_size
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
