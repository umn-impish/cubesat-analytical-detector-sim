import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys; sys.path.append('..')

from adetsim.hafx_src import X123Stack
from adetsim.sim_src import FlareSpectrum

raise ImportError("Import the x123 data loading thing from another file")

def main():
    data = x123_data('C5')
    plot_effective_area(data)

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
