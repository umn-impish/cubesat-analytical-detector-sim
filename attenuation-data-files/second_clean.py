import numpy as np
import sys

'''
attenuation files have repeat values. . . !!!!!!!!!
'''

eng, phot, ray, com = np.loadtxt(sys.argv[1], unpack=True)
rng = np.arange(eng.size)

last_e = -1
for i in rng:
    if eng[i] == last_e:
        eng[i] += 1e-8
    last_e = eng[i]

out = np.array([eng, phot, ray, com])
np.savetxt(sys.argv[1], out.transpose())
