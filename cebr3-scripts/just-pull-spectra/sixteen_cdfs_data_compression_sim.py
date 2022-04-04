import numpy as np
import sys
sys.path.append('..')
from HafxSimulationContainer import HafxSimulationContainer as HSC
from sim_src.FlareSpectrum import FlareSpectrum

print("starting")
thicks = {
    'C1': 0,
    'M1': 60  * 1e-4,
    'M5': 210 * 1e-4,
    'X1': 340 * 1e-4
}

indep_flare_classes = ['B5', 'C5', 'M5', 'X1']

for detector_gc, t in thicks.items():
    print(f"starting {detector_gc}-optimized sims")
    for indep_gc in indep_flare_classes:
        print('indep goes class:', indep_gc)
        fs = FlareSpectrum.make_with_battaglia_scaling(indep_gc, 1.1, 300, 0.1)
        print('got flare spec')
        sc = HSC(t, fs)
        print('initialized container')
        sc.simulate()
        print('done simulating')
        print(sc.matrices[sc.KDISPERSED_RESPONSE])
        att_spec = sc.matrices[sc.KDISPERSED_RESPONSE] @ fs.flare
        np.savetxt(f'{detector_gc}_{indep_gc[0]}', np.array([fs.energies, att_spec]).T)
