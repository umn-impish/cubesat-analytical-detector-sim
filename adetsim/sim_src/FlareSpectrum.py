import astropy.units as u
import numpy as np
# from .sswidl_bridge import battaglia_power_law_with_pivot, f_vth_bridge
import sunxspex.thermal as thermal

GOES_PREFIX = {
    'A': 1e-8,
    'B': 1e-7,
    'C': 1e-6,
    'M': 1e-5,
    'X': 1e-4
}


def goes_class_lookup(flare_specifier: str) -> np.float64:
    ch, mul = flare_specifier[0].upper(), float(flare_specifier[1:])
    return GOES_PREFIX[ch] * mul


def battaglia_power_law_with_pivot(eng_ary, reference_flux, spectral_index, e_pivot):
    # same code as f_1pow
    # just a power law at some reference energy

    # update 16 september 2021: Ethan's code was not consistent with Battaglia model.
    # updated to have spectral index of 1.5 for E < 50 keV
    # assume energyunits are keV
    # update 30 nov 2021: setting break energy to 10 keV just for the hell of it
    BREAK_ENG = 10

    criterion = eng_ary <= BREAK_ENG
    above_break = reference_flux * ((e_pivot / eng_ary) ** spectral_index)

    below_ref = above_break[criterion][-1]
    below_break = below_ref * ((BREAK_ENG / eng_ary) ** 1.5)

    return np.append(below_break[criterion], above_break[np.logical_not(criterion)])


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
    _ENERGY_LIMITS = (1.001, 200.15)
    @classmethod
    def make_with_battaglia_scaling(cls, goes_class: str, start_energy: np.float64,
                                    end_energy: np.float64, de: np.float64, rel_abun=1.0):
        ''' goes flux in W/m2, energies in keV '''
        bp = BattagliaParameters(goes_class_lookup(goes_class))

        edges = np.arange(start_energy, end_energy, step=de)
        energy_centers = edges[:-1] + np.diff(edges)/2

        thermal_edges = edges[
            np.logical_and(
                edges >= FlareSpectrum._ENERGY_LIMITS[0],
                edges <= FlareSpectrum._ENERGY_LIMITS[1]
            )
        ]

        thermal_spec = thermal.thermal_emission(
            thermal_edges * u.keV, bp.plasma_temp * u.MK,
            bp.emission_measure * (u.cm**(-3)),
            abundance_type="sun_coronal",
            observer_distance=(1 * u.AU).to(u.cm)).value

        # make thermal/nonthermal spectra the same size
        thermal_spec = np.append(thermal_spec, np.zeros(energy_centers.size - thermal_spec.size))

        nonthermal_spec = battaglia_power_law_with_pivot(
                energy_centers, bp.reference_flux, bp.spectral_index, bp.reference_energy)    # photon / (s cm2 keV)

        return cls(goes_class, energy_centers, thermal_spec, nonthermal_spec)

    @classmethod
    def dummy(cls, energies: np.ndarray):
        ''' empty spectrum with only energies '''
        return cls('', energies, np.zeros(energies.size), np.zeros(energies.size))

    def __init__(self, goes_class: str, energies: np.ndarray,
                 thermal: np.ndarray, nonthermal: np.ndarray):
        self.goes_class = goes_class
        self.energies = energies
        self.thermal, self.nonthermal = thermal, nonthermal

    @property
    def goes_flux(self) -> np.float64:
        return goes_class_lookup(self.goes_class)

    @property
    def flare(self) -> np.ndarray:
        try:
            return self.thermal + self.nonthermal
        except ValueError:
            print(f"thermal: {self.thermal.size}\tnonthermal: {self.nonthermal.size}")
            raise
