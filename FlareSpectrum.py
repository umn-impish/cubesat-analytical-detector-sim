import numpy as np
from sswidl_bridge import power_law_with_pivot, f_vth_bridge

def goes_class_lookup(flare_specifier: str) -> np.float64:
    lookup = {
        'C' : 1e-6,
        'M' : 1e-5,
        'X' : 1e-6
    }
    ch, mul = flare_specifier[0], float(flare_specifier[1:])
    return lookup[ch] * mul


class BattagliaParameters:
    # see Battaglia et al., https://arxiv.org/abs/astro-ph/0505154v1
    # equations: 11, 12, 8
    def __init__(self, goes_intensity):
        pt_p1 = math.log(goes_intensity)/math.log(10)
        pt_p2 = math.log(3.5) / math.log(10) - 12
        self.plasma_temp = (1 / 0.33) * (pt_p1 - pt_p2)                         # MK

        self.emission_measure = (goes_intensity / 3.6 * 1e50) ** (1/0.92)       # particles^2 / cm3
        self.flux_35kev = (goes_intensity / (1.8e-5)) ** (1/0.83)               # photon / (s cm2 keV)

        if goes_intensity < goes_class_lookup('C2'):
            self.spectral_index = 2.04 * self.flux_35kev**(-0.16)               # unitless
        else:
            self.spectral_index = 3.60 * self.flux_35kev**(-0.16)               # unitless

    def gen_vth_params(self) -> (float, float):
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
    def make_with_battaglia_scaling(goes_flux: np.float64, start_energy: np.float64,
                                    end_energy: np.float64, de: np.float64):
        ''' goes flux in W/m2, energies in keV '''
        bp = BattagliaParameters(goes_intensity)
        pass


    def __init__(self, energies: np.ndarray, flare: np.ndarray):
        self.energies = energies
        self.flare = flare
