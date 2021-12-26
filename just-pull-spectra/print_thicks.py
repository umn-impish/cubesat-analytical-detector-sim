import os
import sys
sys.path.append('..')
from HafxSimulationContainer import HafxSimulationContainer

from pull_consts import from_dir as opt_dir

filez = os.listdir(opt_dir)
for f in sorted(filez, key=lambda x: x[:2]):
    hc = HafxSimulationContainer.from_saved_file(os.path.join(opt_dir, f))
    print(f"{hc.goes_class} attenuator thickness: {hc.al_thick * 1e4:.8f} um")
