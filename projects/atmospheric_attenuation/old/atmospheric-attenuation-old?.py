import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from astropy.table import QTable

import adetsim.sim_src.AttenuationData as ad
from adetsim.sim_src.FlareSpectrum import FlareSpectrum
from adetsim.sim_src.DetectorStack import DetectorStack
from adetsim.sim_src.Material import Material
from adetsim.hafx_src.HafxStack import HafxStack


import plotting
plotting.apply_style('interactive')

# os.makedirs('plots/', exist_ok=True)

OUT_DIR = './atmospheric-attenuation/'
os.makedirs(OUT_DIR, exist_ok=True)


TABLE_PICKLE = '/home/reed/Documents/research/grimpress/cubesat-analytical-detector-sim/atmospheric-attenuation/composition/atmospheric-composition-vs-altitude-full-height-step5.pkl'
with open(TABLE_PICKLE, 'rb') as infile:
    ATMOSPHERIC_LOOKUP_TABLE = pickle.load(infile)


def generate_flare_spectrum(goes_class: str = 'M5') -> FlareSpectrum:

    start, end = 2, 300 # keV
    delta_e = 0.1 # keV
    edges = np.arange(start, end + delta_e, delta_e)

    fs = FlareSpectrum.make_with_battaglia_scaling(
        goes_class=goes_class, 
        energy_edges=edges
    )

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    ax.stairs(fs.nonthermal, fs.energy_edges, label='nonthermal emission')
    ax.stairs(fs.thermal, fs.energy_edges, label='thermal emission')
    ax.stairs(fs.flare, fs.energy_edges, label='whole thing')

    ax.set(
        xlabel='Energy (keV)',
        ylabel='Flare intensity (photons / sec / keV / cm${}^2$)',
        title = f'Incident flare spectrum, GOES {goes_class}',
        xscale='log',
        yscale='log',
        ylim=(1e-4, 1e9)
    )
    ax.legend()
    plt.savefig(os.path.join(OUT_DIR, 'flare-spectrum.png'), dpi=120)

    return fs


def generate_atmospheric_layer(altitude: u.Quantity, thickness: u.Quantity):

    # air_composition = {
    #     'N': 7.81,
    #     'O': 2.09,
    #     'Ar': 0.01,
    #     'C': 0.001,
    # }
    
    # TODO: Renormalize the abundances based on atmoic and molecular elements..........
    row = ATMOSPHERIC_LOOKUP_TABLE[altitude]
    for col in row.columns
    
    elemental_abundances = {}

    layer_attenuation = ad.AttenuationData.from_compound_dict(elemental_abundances)


    layer = Material(DIAMETER, thick, rho, atten_dat, name=name)


def generate_abundance_table():
    return


def attenuate_through_layer():
    return


def main():

    spectrum = generate_flare_spectrum()
    layer = DetectorStack()

    # Units are taken to be in centimeter
    # thin_stack = HafxStack(enable_scintillator=True, att_thick=0)
    # thick_stack = HafxStack(enable_scintillator=True, att_thick=200e-6 * 100)

    # apply_energy_resolution = False
    # thin_response = thin_stack.generate_detector_response_to(fs, disperse_energy=apply_energy_resolution)
    # thick_response = thick_stack.generate_detector_response_to(fs, disperse_energy=apply_energy_resolution)


    # flare_after_thin = thin_response @ fs.flare
    # flare_after_thick = thick_response @ fs.flare

    # fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    # ax.stairs(fs.flare, fs.energy_edges, label='Original flare')
    # ax.stairs(flare_after_thin, fs.energy_edges, label='After thin attenuator')
    # ax.stairs(flare_after_thick, fs.energy_edges, label='After thick attenuator')

    # ax.set(
    #     xlabel='Energy (keV)',
    #     ylabel='(photons or counts) / keV / sec / cm2',
    #     title = '"template" M5 flare',
    #     xscale='log',
    #     yscale='log',
    #     ylim=(1e-6, 1e9)
    # )
    # ax.legend()
    # plt.savefig('./plots/example-detector-sim-with-attenuator.png', dpi=120)
    # plt.show()


if __name__ == '__main__':
    main()