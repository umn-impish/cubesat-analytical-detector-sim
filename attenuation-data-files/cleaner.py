import sys
import numpy as np

'''
output from nist is in the wrong order and energies are in MeV. so let's fix that.
'''

fn = sys.argv[1]
energy_in_mev, rayleigh, compton, phot = np.loadtxt(fn, delimiter='\t', unpack=True)
energy_in_kev = energy_in_mev * 1000
a = np.array([energy_in_kev, phot, rayleigh, compton])
np.savetxt(fn, a.transpose(), delimiter='\t')
