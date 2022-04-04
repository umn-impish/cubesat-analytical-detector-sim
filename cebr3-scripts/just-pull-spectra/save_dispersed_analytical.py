import numpy as np
import os
import sys
sys.path.append('..')
from HafxSimulationContainer import HafxSimulationContainer

from pull_consts import from_dir as pull_from

con_fns = os.listdir(pull_from)
scons = [HafxSimulationContainer.from_saved_file(os.path.join(pull_from, fn)) for fn in con_fns]

for sc in scons:
    e = sc.flare_spectrum.energies
    fl = sc.flare_spectrum.flare
    disp = sc.matrices[sc.KDISPERSED_RESPONSE]
    pure = sc.matrices[sc.KPURE_RESPONSE]

    dispersed_counts = disp @ fl
    gc = sc.flare_spectrum.goes_class
    undisp = pure @ fl

    np.savetxt(f'raw_{gc.lower()}.txt', np.transpose([e, dispersed_counts]))
    print('done with', gc)
