import os
import numpy as np

'''
a whole bunch of constants
'''

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
    # aluminum is special
}

# Directories
AREA_DIR = 'areas'
ATTEN_DIR = 'attenuation-data-files'
DATA_DIR = 'responses-and-areas'
FIG_DIR = 'figures'
LOGS_DIR = 'logs'
SRC_DIR = 'sim_src'

ATTEN_BASENAMES = [AL, TEF, BE, CEBR3]
ATTEN_FILE_FORMAT = "{}.tab"

# attenuation data from:
#   looked here for reference: https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#   took data from here: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
#   nb attenuation data must be in same folder as this file. probably should rework this at some point.
ATTEN_FILES = {
        abn : os.path.join(
        os.path.dirname(__file__),
        ATTEN_DIR,
        ATTEN_FILE_FORMAT.format(abn)) for abn in ATTEN_BASENAMES
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

# energies
MIN_ENG = 1.0               # keV
MAX_ENG = 300.0             # keV
DE = 0.1                    # keV
MIN_THRESHOLD_ENG = 8.0     # keV
MAX_THRESHOLD_ENG = 100.0   # keV

