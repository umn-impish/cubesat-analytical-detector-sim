import sys
import os
import numpy as np
import scipy.integrate
from HafxSimulationContainer import HafxSimulationContainer
import HafxStack

ENG_LOWER_BOUND = 1
ENG_UPPER_BOUND = 100
opt_dir = 'optimized-2021-dec-24'
filez = os.listdir(opt_dir)

base = "{:<30} " * 4

print(base.format("Which", "Flare size", "Al thickness (cm)", f"Counts/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV"))
att_cps_dict = dict()
orig_cps_dict = dict()
thickness_dict = dict()
for f in filez:
    cur_con = HafxSimulationContainer.from_saved_file(os.path.join(opt_dir, f))
    # energies
    en = cur_con.flare_spectrum.energies
    # cur_con.simulate()
    # attenuated flare spectrum
    att = np.matmul(cur_con.matrices[cur_con.KDISPERSED_RESPONSE], cur_con.flare_spectrum.flare)

    restrict = np.logical_and(en > ENG_LOWER_BOUND, en < ENG_UPPER_BOUND)

    att_cps = scipy.integrate.simpson(att[restrict], x=en[restrict]) * HafxStack.SINGLE_DET_AREA
    orig_cps = scipy.integrate.simpson(cur_con.flare_spectrum.flare[restrict], x=en[restrict]) * HafxStack.SINGLE_DET_AREA

    fl_sz = cur_con.flare_spectrum.goes_class
    att_cps_dict[fl_sz] = att_cps
    orig_cps_dict[fl_sz] = orig_cps
    thickness_dict[fl_sz] = cur_con.al_thick

for k in sorted(att_cps_dict.keys(), key=lambda x: ord(x[0]) + int(x[1])):
    line_fmt = "{:<30} {:<30} {:<30.6e} {:<30.0f}"
    print(line_fmt.format('original', k, thickness_dict[k], orig_cps_dict[k]))
    print(line_fmt.format('attenuated', k, thickness_dict[k], att_cps_dict[k]))
