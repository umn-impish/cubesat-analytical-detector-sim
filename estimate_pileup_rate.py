import numpy as np
import os
import scipy.integrate

import sim_src.impress_constants as ic

def pileup_fraction_estimate(count_rate, pileup_time):
    # waiting time probability density function
    def pdf(t): return count_rate * np.exp(-count_rate * t)
    return scipy.integrate.quad(pdf, 0, pileup_time)

opt_id = "optimized"
pileup_time = 0.75 * 1e-6       # microsecond
opt_files = [f for f in os.listdir(ic.DATA_DIR) if opt_id in f]

loaded = dict()
for f in opt_files:
    parts = f.split('_')
    size = parts[1]
    loaded[size] = np.load(os.path.join(ic.DATA_DIR, f))

keyz = ('C1', 'C5', 'M1', 'M5', 'X1')
for k in keyz:
    v = loaded[k]
    fs = v[ic.FS_KEY]
    eng = v[ic.ENG_KEY]
    resp = v[ic.RESP_KEY]
    att = np.matmul(resp, fs)
    count_rate = scipy.integrate.simpson(att, x=eng) * ic.SINGLE_DET_AREA
    frac, err = pileup_fraction_estimate(count_rate, pileup_time)
    print(f"{k}, attenuator window t = {v[ic.THICK_KEY]:.3e} cm, pileup fraction estimate: {frac:.3f} +- {err:.3e}")
