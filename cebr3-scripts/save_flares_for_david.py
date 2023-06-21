import json
import numpy as np
import os

import adetsim.sim_src.FlareSpectrum as fs
import adetsim.hafx_src.HafxStack as hs

flare_dat = (
    ('C1', 0),
    ('M1', 50),
    ('M5', 260),
    ('X1', 550)
)
edges = np.arange(1.1, 300, 0.1)

out_dir = 'for-david'
os.makedirs(out_dir, exist_ok=True)
for (cl, thk) in flare_dat:
    bp = fs.BattagliaParameters(fs.goes_class_lookup(cl))
    print(bp)
    fl = fs.FlareSpectrum.make_with_battaglia_scaling(
        goes_class=cl,
        energy_edges=edges
    )
    with open(f'{out_dir}/{cl.lower()}.json', 'w') as f:
        out = {
            'energy_edges': list(edges),
            'ct/sec/keV/cm2': list(fl.flare)
        }
        json.dump(out, f, indent=2, sort_keys=True)


    st = hs.HafxStack(att_thick=thk/1e4)
    srm = st.generate_detector_response_to(fl, disperse_energy=False)
    att = srm @ fl.flare
    with open(f'{out_dir}/{cl.lower()}-attenuated.json', 'w') as f:
        out = {
            'energy_edges': list(edges),
            'ct/sec/keV/cm2': list(att)
        }
        json.dump(out, f, indent=2, sort_keys=True)
