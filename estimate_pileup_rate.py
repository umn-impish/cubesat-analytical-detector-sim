import numpy as np
import os
import scipy.integrate
import sim_src.impress_constants as ic
from HafxSimulationContainer import HafxSimulationContainer

def pileup_fraction_estimate(count_rate, pileup_time):
    # waiting time probability density function
    def pdf(t):
        return count_rate * np.exp(-count_rate * t)
    return scipy.integrate.quad(pdf, 0, pileup_time)

optim_dir = 'optimized-2-aug-2021'
pileup_time = 0.75 * 1e-6       # microsecond
opt_files = os.listdir(optim_dir)

loaded = dict()
for f in opt_files:
    con = HafxSimulationContainer.from_saved_file(os.path.join(optim_dir, f))
    loaded[con.flare_spectrum.goes_class] = con

keyz = ('C1', 'C5', 'M1', 'M5', 'X1')
cols = ("Flare size", "Attenuator thickness (cm)", "Pileup fraction estimate", "Estimate error")
col_str = ("{:<30}" * len(cols)).format(*cols)
print(col_str)
for k in keyz:
    # doesn't matter if we use the energy-dispersed matrix or not
    # total counts are conserved
    threshold_energy = 1            # keV
    cur_con = loaded[k]
    disp_flare = np.matmul(cur_con.matrices[cur_con.KPURE_RESPONSE], cur_con.flare_spectrum.flare)
    eng = cur_con.flare_spectrum.energies
    count_rate = scipy.integrate.simpson(
            disp_flare[eng >= threshold_energy], x=eng[eng >= threshold_energy]) * ic.SINGLE_DET_AREA
    frac, err = pileup_fraction_estimate(count_rate, pileup_time)
    err_str = f"{(err * 100):.2e}%"
    print(f"{k:<30}{cur_con.al_thick:<30.3e}{frac:<30.2%}+-{err_str:<30}")
