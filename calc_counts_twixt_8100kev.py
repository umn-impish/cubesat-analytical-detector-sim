import sys
import os
import numpy as np
from scipy.integrate import simpson
import sim_src.impress_constants as ic

opt_dir = 'optimized'
resp_dir = os.path.join(ic.DATA_DIR, opt_dir)
filez = os.listdir(resp_dir)
filez.sort(key=lambda x: x[:2])

base = "{:<30} {:<30} {:<30}"

print(base.format("Flare size", "Al thickness (cm)", "Counts/sec for energy in (8, 100) keV"))
for f in filez:
    with np.load(os.path.join(resp_dir, f)) as dat:
        en = dat[ic.ENG_KEY]
        spec = dat[ic.FS_KEY]
        res = dat[ic.RESP_KEY]
        thick = dat[ic.THICK_KEY]
    att = np.matmul(res, spec) * ic.FULL_AREA / 4
    restrict = np.logical_and(en > 8, en < 100)
    total_cps = simpson(att[restrict], x=en[restrict])
    fl_sz = f.split('_')[0]
    print(f"{fl_sz:<30} {thick:<30.6e} {total_cps:<30.0f}")
