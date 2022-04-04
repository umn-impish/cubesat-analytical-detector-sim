import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys; sys.path.append('..')

from adetsim.hafx_src import X123Stack
from adetsim.sim_src import FlareSpectrum

def main():
    data = x123_data('C5')
    plot_effective_area(data)

def x123_data(goes_class):
    save_fn = f'{goes_class}-x123-resp-saved.xz'
    if os.path.exists(save_fn) and goes_class in save_fn:
        print('load pickle')
        with lzma.open(save_fn, 'rb') as f:
            dat = pickle.load(f)
    else:
        print('make from scratch')
        dat = dict()
        fs = FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
            goes_class=goes_class, start_energy=1.1, end_energy=400, de=0.1)
        dat['energies'] = fs.energies
        dat['flare'] = fs.flare
        print('done flare spectrum')

        xs = X123Stack.X123Stack()
        dat['area'] = xs.area
        dat['disp_resp'] = (disp_resp := xs.generate_detector_response_to(fs, disperse_energy=True))
        print('done dispersed')
        dat['undisp_resp'] = (undisp_resp := xs.generate_detector_response_to(fs, disperse_energy=False))
        print('done undispersed')

        dat['disp'] = disp_resp @ fs.flare
        print('done disp multiply')
        dat['undisp'] = undisp_resp @ fs.flare
        print('done undisp multiply')

        with lzma.open(save_fn, 'wb') as f:
            pickle.dump(dat, f)
        print('done pickle')

    return dat

def plot_effective_area(dat):
    print('area is', dat['area'])
    area_vec = dat['area'] * np.ones_like(dat['energies'])
    effective_area = dat['undisp_resp'] @ area_vec
    fig, ax = plt.subplots()
    ax.plot(dat['energies'], effective_area, label='X-123 effective area')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Area (cm${}^2$)')
    ax.set_title('Effective area')
    fig.tight_layout()
    plt.show()

def plot_spectra(dat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dat['energies'], dat['area'] * dat['flare'], label='Flare spectrum')
    ax.plot(dat['energies'], dat['area'] * dat['undisp'], label='X-123 spectrum (no energy resolution applied)')
    ax.plot(dat['energies'], dat['area'] * dat['disp'], label='X-123 spectrum (energy resolution applied)')

    ax.set_xlim(1, max(dat['energies']))
    ax.set_ylim(1e-3, 1e9)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('(photons or counts) / keV / sec')
    fig.tight_layout()
    plt.show()
    print('done plot')

if __name__ == '__main__': main()
