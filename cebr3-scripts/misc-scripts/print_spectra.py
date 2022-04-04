import numpy as np
import os
from HafxSimulationContainer import HafxSimulationContainer

d = 'optimized-10-aug-2021'

containers = []
with open('william_spectra.py', 'w') as write_file:
    print("import numpy as np\n", file=write_file)
    print("energies = np.arange({:.1f}, {:.1f}, {:.2f})\n".format(
        HafxSimulationContainer.MIN_ENG,
        HafxSimulationContainer.MAX_ENG + HafxSimulationContainer.DE,
        HafxSimulationContainer.DE), file=write_file)

    for read_file in os.listdir(d):
        c = HafxSimulationContainer.from_saved_file(os.path.join(d, read_file))
        energies = c.flare_spectrum.energies
        att_spec = np.matmul(c.matrices[c.KDISPERSED_RESPONSE], c.flare_spectrum.flare)
        print(f"{c.flare_spectrum.goes_class.lower()}_flare = [", file=write_file)
        STEP = 6
        for i in range(0, len(att_spec), STEP):
            fmt = "{:.3e}, " * np.min((STEP, len(att_spec) - i))
            prnt = fmt.format(*att_spec[i:i+STEP])
            print(prnt, file=write_file)
        print("]\n", file=write_file)
