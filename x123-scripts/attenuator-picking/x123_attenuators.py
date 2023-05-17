import astropy.units as u
import lzma
import numpy as np
import os
import pickle

from adetsim.sim_src import FlareSpectrum
from adetsim.sim_src.Material import Material
from adetsim.sim_src.AttenuationData import AttenuationData

import adetsim.hafx_src.HafxMaterialProperties as hmp
from adetsim.hafx_src import X123Stack

out_fn = 'x123-processed-attenuators.pkl.lzma'
edges = np.arange(3.5, 20, 0.02)

if __name__ == '__main__':
    trial_thicks = [(t << u.um).to(u.cm).value for t in range(10, 100, 20)]

    classes = ('C1', 'M1', 'M5', 'X1')
    flares = [
        FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc, energy_edges=edges
        )
        for gc in classes
    ]

    if not os.path.exists(out_fn):
        processed = {cl: dict() for cl in classes}

        for thk in trial_thicks:
            print('start', thk*1e4, 'um thick attenuator')
            xs = X123Stack.X123Stack()
            attenuator = Material(
                diameter=xs.materials[0].diameter,
                attenuation_thickness=thk,
                mass_density=hmp.DENSITIES[hmp.AL],
                attenuation_data=AttenuationData.from_nist_file(hmp.ATTEN_FILES[hmp.AL]),
                name='Al'
            )

            xs.materials.insert(0, attenuator)
            res = xs.generate_detector_response_to(flares[0], disperse_energy=True)
            for fl in flares:
                print('\tdo', fl.goes_class)
                processed[fl.goes_class][f'{thk*1e4:.0f}um'] = res @ fl.flare

        with lzma.open(out_fn, 'wb') as f:
            pickle.dump(processed, f)
