import sys
import os
import numpy as np
from scipy.integrate import simpson
import sim_src.impress_constants as ic
from HafxSimulationContainer import HafxSimulationContainer

opt_dir = 'optimized-2-aug-2021'
filez = os.listdir(opt_dir)
filez.sort(key=lambda x: x.split('_')[-3])

base = "{:<30} {:<30} {:<30}"

print(base.format("Flare size", "Al thickness (cm)", "Counts/sec for energy in (8, 100) keV"))
for f in filez:
    cur_con = HafxSimulationContainer.from_saved_file(os.path.join(opt_dir, f))
    # energies
    en = cur_con.flare_spectrum.energies
    # attenuated flare spectrum
    att = np.matmul(cur_con.matrices[cur_con.KDISPERSED_RESPONSE], cur_con.flare_spectrum.flare)

    restrict = np.logical_and(en > 8, en < 100)
    total_cps = simpson(att[restrict], x=en[restrict])
    fl_sz = cur_con.flare_spectrum.goes_class
    print(f"{fl_sz:<30} {cur_con.al_thick:<30.6e} {total_cps:<30.0f}")
