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
BE_THICKNESS = 0.075    # cm
TEFLON_THICKNESS = 0.0127  # cm
CEBR3_THICKNESS = 0.5   # cm
THICKNESSES = {
    BE : BE_THICKNESS,
    TE : TEFLON_THICKNESS,
    CEBR3 : CEBR3_THICKNESS
    # aluminum is special
}

ATTEN_FOLDER = 'attenuation-data-files'
ATTEN_BASENAMES = [AL, TE, BE, CEBR3]
ATTEN_FILE_FORMAT = "{}.tab"

# dict, or hashmap
ATTEN_FILES = { abn : os.path.join(
    ATTEN_FOLDER,
    ATTEN_FILE_FORMAT.format(abn)) for abn in ATTEN_BASENAMES }

RHO_AL = 2.70       # g / cm3
RHO_BE = 1.848      # g / cm3
RHO_TEF = 2.2       # g / cm3
RHO_CEBR3 = 5.1     # g / cm3
DENSITIES = {
    AL : RHO_AL,
    BE : RHO_BE,
    TE : RHO_TEF,
    CEBR3 : RHO_CEBR3
}

diameter = np.sqrt((43 / 4) / np.pi / 2)   # cm
