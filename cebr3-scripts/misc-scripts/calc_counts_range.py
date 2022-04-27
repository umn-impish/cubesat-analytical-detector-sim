import sys
import os
import numpy as np
import scipy.integrate
sys.path.insert(0, '../..')
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from adetsim.hafx_src.HafxStack import HafxStack

ENG_LOWER_BOUND = 11
ENG_UPPER_BOUND = 26
opt_dir = f'{os.path.dirname(__file__)}/../../responses-and-areas/optimized-2021-dec-24'
filez = os.listdir(opt_dir)

base = "{:<30} " * 4

print(base.format("Which", "Flare size", "Al thickness (cm)", f"count/cm2/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV"))
att_cps_cm2_d = dict()
orig_cps_cm2_d = dict()
thickness_dict = dict()
for f in filez:
    cur_con = HafxSimulationContainer.from_saved_file(os.path.join(opt_dir, f))
    fl = cur_con.flare_spectrum.flare
    en = cur_con.flare_spectrum.energies
    att_fl = np.matmul(
        cur_con.matrices[cur_con.KDISPERSED_RESPONSE],
        fl)

    restrict = np.logical_and(en >= ENG_LOWER_BOUND, en <= ENG_UPPER_BOUND)

    # value * bin width
    att_counts_cm2_sec = np.sum(att_fl[restrict] * 1)#np.diff(en)[0]) #scipy.integrate.simpson(att_fl[restrict], x=en[restrict]) 
    orig_counts_cm2_sec = np.sum(fl[restrict] * 1)#np.diff(en)[0]) #scipy.integrate.simpson(fl[restrict], x=en[restrict]) 

    fl_sz = cur_con.flare_spectrum.goes_class
    att_cps_cm2_d[fl_sz] = att_counts_cm2_sec
    orig_cps_cm2_d[fl_sz] = orig_counts_cm2_sec
    thickness_dict[fl_sz] = cur_con.al_thick

for k in sorted(att_cps_cm2_d.keys(), key=lambda x: ord(x[0]) + int(x[1])):
    line_fmt = "{:<30} {:<30} {:<30.6e} {:<30.0f}"
    print(line_fmt.format('original', k, thickness_dict[k], orig_cps_cm2_d[k]))
    print(line_fmt.format('attenuated', k, thickness_dict[k], att_cps_cm2_d[k]))
