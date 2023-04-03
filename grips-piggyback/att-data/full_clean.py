import os
import numpy as np

''' clean the raw nist data and make it happy '''

raw_data_dir = 'nist'
out_dir = 'cleaned'

for fn in os.listdir(raw_data_dir):
    raw_fn = os.path.join(raw_data_dir, fn)
    if os.path.isfile(fn):
        raise OSError(f"File {fn} exists already. Delete it and try again (make sure you're in the right directory")
    eng_mev, ray, com, phot = np.loadtxt(raw_fn, unpack=True)
    # switch to keV from NIST MeV
    eng = eng_mev * 1000
    last_e = -1
    for i in range(eng.size):
        if eng[i] == last_e:
            # add on a tiny bit so that the energies aren't exactly the same
            eng[i] += 1e-8
        last_e = eng[i]
    out = np.array([eng, phot, ray, com])
    np.savetxt(os.path.join(out_dir, fn), out.transpose())
