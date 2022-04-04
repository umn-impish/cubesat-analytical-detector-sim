import os

''' a bunch of constants '''
AL = 'Al'
TEF = 'Teflon'
BE = 'Be'
CEBR3 = 'CeBr3'
SI = 'Si'
# order on HaFX detector
HAFX_MATERIAL_ORDER = [AL, TEF, BE, CEBR3]
HAFX_DEAD_TIME = 0.5e-6     # s

'''
NB: these all need to get re-verified. i just took them from Ethan's code.
update 24 sep 2021: updated to match Geant sims (teflon is like 10x thicker than I had. whoops)
update 28 sep 2021: went back to old thickness; G4 thickness was ridiculous and overattenuating
'''
BE_THICKNESS = 0.07         # cm
TEFLON_THICKNESS = 0.0127   # cm
CEBR3_THICKNESS = 0.5       # cm
THICKNESSES = {
    BE: BE_THICKNESS,
    TEF: TEFLON_THICKNESS,
    CEBR3: CEBR3_THICKNESS
    # aluminum is special so we leave it out
}

ATTEN_DIR = 'all-attenuation-data/attenuation-data-files'
ATTEN_BASENAMES = [AL, TEF, BE, CEBR3, SI]

# attenuation data from:
#   looked here for reference: https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#   took data from here: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
#   nb attenuation data must be in same folder as this file. probably should rework this at some point. (lol not gonna happen, sep2021)
ATTEN_FILES = {
    abn: os.path.join(
        os.path.dirname(__file__),
        ATTEN_DIR,
        f"{abn}.tab") for abn in ATTEN_BASENAMES
}

RHO_AL = 2.712      # g / cm3
RHO_BE = 1.850      # g / cm3
RHO_TEF = 2.200     # g / cm3
RHO_CEBR3 = 5.1     # g / cm3
RHO_SI = 2.33       # g / cm3
DENSITIES = {
    AL: RHO_AL,
    BE: RHO_BE,
    TEF: RHO_TEF,
    CEBR3: RHO_CEBR3,
    SI: RHO_SI
}

FULL_AREA = 43                                  # cm2 
SINGLE_DET_AREA = FULL_AREA / 4                 # cm2
DIAMETER = 3.7                                  # cm
METER_PER_CM = 1e2                              # cm per meter
