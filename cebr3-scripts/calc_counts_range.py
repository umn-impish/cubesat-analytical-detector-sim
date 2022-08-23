import os
import numpy as np
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer

ENG_LOWER_BOUND = 14
ENG_UPPER_BOUND = 300
CEBR3_AREA = 43/4
opt_dir = f'{os.path.dirname(__file__)}/../responses-and-areas/optimized-2022-aug-22-bins'
filez = os.listdir(opt_dir)

base = "{:<30} " * 4

print(base.format("Which", "Flare size", "Al thickness (um)", f"count/cm2/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV"))
att_cps_cm2_d = dict()
orig_cps_cm2_d = dict()
thickness_dict = dict()
for f in filez:
    cur_con = HafxSimulationContainer.from_saved_file(os.path.join(opt_dir, f))
    fl = cur_con.flare_spectrum.flare
    en = cur_con.flare_spectrum.energy_edges
    att_fl = np.matmul(
        cur_con.matrices[cur_con.KDISPERSED_RESPONSE],
        fl)

    restrict = np.logical_and(en >= ENG_LOWER_BOUND, en <= ENG_UPPER_BOUND)

    # value * bin width
    att_counts_cm2_sec = np.sum((att_fl * np.diff(en))[restrict[:-1]])
    orig_counts_cm2_sec = np.sum((fl * np.diff(en))[restrict[:-1]])

    fl_sz = cur_con.flare_spectrum.goes_class
    att_cps_cm2_d[fl_sz] = att_counts_cm2_sec
    orig_cps_cm2_d[fl_sz] = orig_counts_cm2_sec
    thickness_dict[fl_sz] = cur_con.al_thick * 1e4

line_fmt = "{:<30} {:<30} {:<30.2f} {:<30.0f}"
for k in sorted(att_cps_cm2_d.keys(), key=lambda x: ord(x[0]) + int(x[1])):
    print(line_fmt.format('original', k, thickness_dict[k], orig_cps_cm2_d[k]))
    print(line_fmt.format('attenuated', k, thickness_dict[k], att_cps_cm2_d[k]))

print()
print(base.format("Which", "Flare size", "Al thickness (um)", f"count/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV"))
for k in sorted(att_cps_cm2_d.keys(), key=lambda x: ord(x[0]) + int(x[1])):
    print(line_fmt.format('original', k, thickness_dict[k], orig_cps_cm2_d[k] * CEBR3_AREA))
    print(line_fmt.format('attenuated', k, thickness_dict[k], att_cps_cm2_d[k] * CEBR3_AREA))
