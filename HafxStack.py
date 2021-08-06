import numpy as np
import os

from sim_src.AttenuationData import AttenuationData, AttenuationType
from sim_src.DetectorStack import DetectorStack
from sim_src.FlareSpectrum import FlareSpectrum
from sim_src.Material import Material
from Sipm3000 import Sipm3000

''' a bunch of constants '''
AL = 'Al'
TEF = 'Teflon'
BE = 'Be'
CEBR3 = 'CeBr3'
# order on HaFX detector
HAFX_MATERIAL_ORDER = [AL, TEF, BE, CEBR3]
HAFX_DEAD_TIME = 1e-6       # s

'''
NB: these all need to get re-verified. i just took them from Ethan's code.
'''
BE_THICKNESS = 0.075        # cm
TEFLON_THICKNESS = 0.0127   # cm
CEBR3_THICKNESS = 0.5       # cm
THICKNESSES = {
    BE : BE_THICKNESS,
    TEF : TEFLON_THICKNESS,
    CEBR3 : CEBR3_THICKNESS
    # aluminum is special so we leave it out
}

ATTEN_DIR = 'all-attenuation-data/attenuation-data-files'
ATTEN_BASENAMES = [AL, TEF, BE, CEBR3]

# attenuation data from:
#   looked here for reference: https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#   took data from here: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
#   nb attenuation data must be in same folder as this file. probably should rework this at some point.
ATTEN_FILES = {
    abn : os.path.join(
        os.path.dirname(__file__),
        ATTEN_DIR,
        f"{abn}.tab") for abn in ATTEN_BASENAMES
}

RHO_AL = 2.699      # g / cm3
RHO_BE = 1.848      # g / cm3
RHO_TEF = 2.250     # g / cm3
RHO_CEBR3 = 5.1     # g / cm3
DENSITIES = {
    AL : RHO_AL,
    BE : RHO_BE,
    TEF : RHO_TEF,
    CEBR3 : RHO_CEBR3
}

FULL_AREA = 43                                  # cm2 
SINGLE_DET_AREA = FULL_AREA / 4                 # cm2
DIAMETER = 2 * np.sqrt(FULL_AREA / 4 / np.pi)   # cm


def gen_materials(al_thick: np.float64):
    ''' put the HaFX materials in the right order (variable aluminum thickness)'''
    mat_order = HAFX_MATERIAL_ORDER
    materials = []
    for name in mat_order:
        thick = al_thick if name == AL else THICKNESSES[name]
        rho = DENSITIES[name]
        atten_dat = AttenuationData.from_nist_file(ATTEN_FILES[name])
        materials.append(Material(DIAMETER, thick, rho, atten_dat))
    return materials


class HafxStack(DetectorStack):
    ''' photoabsorption into the scintillator crystal is different here so we need separate behavior. '''
    def __init__(self, enable_scintillator: bool=True, al_thick: np.float64=NotImplemented):
        super().__init__(gen_materials(al_thick), Sipm3000())
        # take off the scintillator to treat it separately
        self.scintillator = self.materials.pop()
        # XXX: set to True to disable the scintillator (i.e. only disperse spectrum, dont absorb it)
        self.enable_scintillator = enable_scintillator

    def generate_detector_response_to(
            self, incident_spectrum: FlareSpectrum, disperse_energy: bool, chosen_attenuations: list=AttenuationType.ALL) -> np.ndarray:
        response = self._generate_material_response_due_to(incident_spectrum, chosen_attenuations)
        if self.enable_scintillator:
            # now incorporate the scintillator
            absorbed = self.generate_scintillator_response(incident_spectrum, chosen_attenuations)
            response = np.matmul(absorbed, response)
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

    def generate_scintillator_response(self, incident_spectrum: FlareSpectrum, chosen_attenuations: list) -> np.ndarray:
        ident = np.identity(incident_spectrum.energies.size)
        # we must include photoelectric absorption as this mechanism leads to scintillation.
        abs_atts = list(set([AttenuationType.PHOTOELECTRIC_ABSORPTION] + chosen_attenuations))
        # XXX: only dimensions of incident_spectrum used in this call. confusing...
        absorbed = ident - self.scintillator.generate_overall_response_matrix_given(incident_spectrum, abs_atts)
        return absorbed
