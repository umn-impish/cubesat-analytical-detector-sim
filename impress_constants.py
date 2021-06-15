import os
import numpy as np

'''
a whole bunch of constants
'''

AL = 'Al'
TE = 'Teflon'
BE = 'Be'
CEBR3 = 'CeBr3'

'''
NB: these all need to get re-verified. i just took them from Ethan's code.
'''
BE_THICKNESS = 0.075        # cm
TEFLON_THICKNESS = 0.0127   # cm
CEBR3_THICKNESS = 0.5       # cm
THICKNESSES = {
    BE : BE_THICKNESS,
    TE : TEFLON_THICKNESS,
    CEBR3 : CEBR3_THICKNESS
    # aluminum is special
}

ATTEN_FOLDER = 'attenuation-data-files'
ATTEN_BASENAMES = [AL, TE, BE, CEBR3]
ATTEN_FILE_FORMAT = "{}.tab"

# attenuation data from:
#   looked here for reference: https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#   took data from here: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
ATTEN_FILES = {
        abn : os.path.join(
        ATTEN_FOLDER,
        ATTEN_FILE_FORMAT.format(abn)) for abn in ATTEN_BASENAMES }

RHO_AL = 2.699      # g / cm3
RHO_BE = 1.848      # g / cm3
RHO_TEF = 2.250     # g / cm3
RHO_CEBR3 = 5.1     # g / cm3
DENSITIES = {
    AL : RHO_AL,
    BE : RHO_BE,
    TE : RHO_TEF,
    CEBR3 : RHO_CEBR3
}

FULL_AREA = 43                                  # cm2 
DIAMETER = 2 * np.sqrt(FULL_AREA / 4 / np.pi)   # cm

ENG_KEY = 'energies'
RESP_KEY = 'response_matrix'
EFFA_KEY = 'eff_area'
FS_KEY = 'flare_spectrum'

# Directories
DATA_DIR = 'responses-and-areas'
FIG_DIR = 'figures'
AREA_DIR = 'areas'
LOGS_DIR = 'logs'
