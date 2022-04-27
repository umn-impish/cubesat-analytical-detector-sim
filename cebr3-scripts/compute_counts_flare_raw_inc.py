import sys; sys.path.append('..')
import numpy as np
from adetsim.sim_src.FlareSpectrum import FlareSpectrum

goes_class = 'X1'
start_energy = 1
end_energy = 100

fs = FlareSpectrum.make_with_battaglia_scaling(
    goes_class, 1.1, 300, 0.01)

eng_cut = (fs.energies >= start_energy) & (fs.energies <= end_energy)

e, f = fs.energies[eng_cut], fs.flare[eng_cut]

countrate = np.trapz(f, x=e)
print(f'count rate for {goes_class} flare from {start_energy} to {end_energy} is: {countrate:.3e} phot/sec/cm2')
