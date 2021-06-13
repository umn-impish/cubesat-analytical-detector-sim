import numpy as np
import os
import sys
# hack, don't do this usually but i'm lazy so whatever
sys.path.append(os.path.abspath('..'))
from PhotonDetector import gaussian_row

# simulate energy numpy array
DELTA_E = 0.4
step_size = 0.1
engs = np.arange(1, 300, step=step_size)
# number of rows
dim = engs.size

fwhm = DELTA_E * engs / step_size
sd = fwhm / (2 * np.sqrt(2 * np.log(2)))
verified = np.empty((dim, dim))
rng = np.arange(dim)
for i in rng:
    # make each matrix row a Gaussian centered at the diagonal
    verified[i] = gaussian_row(dim, sd, i)

vectorized_indices = np.tile(rng, (dim, 1)).transpose()
vectorized = gaussian_row(dim, sd, vectorized_indices)

if np.array_equal(verified, vectorized):
    print("pass")
else:
    print("fail")
