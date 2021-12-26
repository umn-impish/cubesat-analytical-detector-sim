import numpy as np
import os
import sys
sys.path.append('..')
from HafxSimulationContainer import HafxSimulationContainer

from pull_consts import from_dir as pull_from

num_photons = 1000000

con_fns = os.listdir(pull_from)
scons = [HafxSimulationContainer.from_saved_file(os.path.join(pull_from, fn)) for fn in con_fns]

for sc in scons:
    e = sc.flare_spectrum.energies
    fl = sc.flare_spectrum.flare
    pure = sc.matrices[sc.KPURE_RESPONSE]

    photons_incident = num_photons / e.size * np.ones_like(e)
    print("photons in one entry:", photons_incident[0])
    dispersed_photons = np.matmul(pure, photons_incident)
    print(f"total dispersed photons: {np.sum(dispersed_photons)}")

    gc = sc.flare_spectrum.goes_class
    unatt = np.matmul(pure, fl)

    np.savetxt(f'{gc}-analytical-unatt.txt', np.transpose([e, unatt]))
    np.savetxt(f'{gc}-{num_photons}-phots.txt', np.transpose([e, dispersed_photons]))
