import numpy as np
from .LinearInterpolateDisperseDetector import gaussian_row
from ..sim_src import PhotonDetector
from ..sim_src.FlareSpectrum import FlareSpectrum

class FixedEnergyResolutionDetector(PhotonDetector.PhotonDetector):
    # conservative estimate of energy resolution at 5.9 keV
    ERES = 0.03
    def __init__(self, eres=ERES):
        super().__init__()

    def generate_energy_resolution_given(self, spectrum) -> np.ndarray:
        de = spectrum.energies[1] - spectrum.energies[0]
        fwhm = self.ERES * spectrum.energies / de

        dim = spectrum.energies.size
        idxs = np.arange(dim)
        vec_idxs = np.tile(idxs, (dim, 1)).transpose()

        return gaussian_row(dim, fwhm, vec_idxs)
