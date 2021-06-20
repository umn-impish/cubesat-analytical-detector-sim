import numpy as np
from .sswidl_bridge import power_law_with_pivot, f_vth_bridge

GOES_PREFIX = {
    'C' : 1e-6,
    'M' : 1e-5,
    'X' : 1e-4
}

def goes_class_lookup(flare_specifier: str) -> np.float64:
    ch, mul = flare_specifier[0].upper(), float(flare_specifier[1:])
    return GOES_PREFIX[ch] * mul


class BattagliaParameters:
    # see Battaglia et al., https://arxiv.org/abs/astro-ph/0505154v1
    # equations: 11, 12, 8
    def __init__(self, goes_flux):
        pt_p1 = np.log(goes_flux)/np.log(10)
        pt_p2 = np.log(3.5) / np.log(10) - 12
        self.plasma_temp = (1 / 0.33) * (pt_p1 - pt_p2)                         # MK

        self.emission_measure = (goes_flux / 3.6 * 1e50) ** (1/0.92)            # particles^2 / cm3
        self.reference_energy = 35.0                                            # keV
        self.reference_flux = (goes_flux / (1.8e-5)) ** (1/0.83)                # photon / (s cm2 keV)

        if goes_flux < goes_class_lookup('C2'):
            self.spectral_index = 2.04 * self.reference_flux**(-0.16)           # unitless
        else:
            self.spectral_index = 3.60 * self.reference_flux**(-0.16)           # unitless

    def gen_vth_params(self) -> (float, float):
        K_B = 8.627e-8  # keV/K
        '''
        generate plasma temperature, emission measure in units appropriate for idl function f_vth.
        returns: (plasma_temp, emission_measure) in units of (keV, 1e49 * particles2 / cm3)
        '''
        pt = self.plasma_temp * K_B * 1e6
        em = self.emission_measure / 1e49
        return (pt, em)


class FlareSpectrum:
    # NB: this might not be the best way to organize this, it's just how i first thought to do it
    @classmethod
    def make_with_battaglia_scaling(cls, goes_class: str, start_energy: np.float64,
            end_energy: np.float64, de: np.float64, rel_abun: np.float64 = 1.0):
        ''' goes flux in W/m2, energies in keV '''
        bp = BattagliaParameters(goes_class_lookup(goes_class))
        energies = np.arange(start_energy, end_energy, de)                              # keV
        good_pt, good_em = bp.gen_vth_params()                                          # (keV, 1e49particle2 / cm3)
        thermal_spec = f_vth_bridge(
                start_energy, end_energy, de, good_em, good_pt, rel_abun)               # photon / (s cm2 keV)
        nonthermal_spec = power_law_with_pivot(
                energies, bp.reference_flux, bp.spectral_index, bp.reference_energy)    # photon / (s cm2 keV)

        return cls(goes_class, energies, thermal_spec, nonthermal_spec)

    @classmethod
    def dummy(cls, energies: np.ndarray):
       return cls('', energies, np.zeros(energies.size), np.zeros(energies.size))

    def __init__(self, goes_class: str, energies: np.ndarray,
                 thermal: np.ndarray, nonthermal: np.ndarray):
        self.goes_class = goes_class
        self.energies = energies
        self.flare = thermal + nonthermal

    @property
    def goes_flux(self) -> np.float64:
        return goes_class_lookup(self.goes_class)
#     @property
#     def flare(self) -> np.ndarray:
#         return self.thermal + self.nonthermal
