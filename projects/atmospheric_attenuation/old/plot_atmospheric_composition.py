# import argparse
# import astropy.units as u
# import itertools
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pickle

# from astropy.table import QTable

# import plotting
# plotting.apply_style('interactive')


# def read_data(pickle_file: str) -> QTable:
#     with open(pickle_file, 'rb') as infile:
#         return pickle.load(infile)


# def plot_densities(data: QTable) -> plt.Axes:

#     fig, ax = plt.subplots(figsize=(8,6), layout='constrained', sharex=True)

#     altitude = data['Heit']
#     marker = itertools.cycle(('x', '+', 'o', '*', 'v', 's'))
#     for column in data.columns:
#         if 'den' in column and isinstance(data[column], u.Quantity):
#             label = column.replace('den', '')
#             ax.plot(altitude, data[column], marker=next(marker), ls='--', markersize=4, lw=0.25, label=label)
#     ax.plot(altitude, data['air'], marker=next(marker), ls='--', markersize=3, lw=0.25, label='air')
#     ax.axvline(40, c='gray', ls=':')

#     ax.set(
#         xlabel=f'Altitude ({altitude.unit})',
#         ylabel=f'Density ({data[column].unit})',
#         yscale='log',
#         ylim=(1e-20, ax.get_ylim()[1])
#     )
#     ax.legend()

#     return ax


# def plot_abundances(data: QTable, ax: plt.Axes = None) -> plt.Axes:
    
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8,6), layout='constrained', sharex=True)
    
#     altitude = data['altitude']
#     marker = itertools.cycle(('x', '+', 'o', '*', 'v', 's', 'h', 'D'))

#     for column in data.columns:
#         if 'abund' in column:
#             label = column.replace('abund', '')
#             ax.plot(altitude, data[column], marker=next(marker), ls='--', markersize=2, lw=0.3, label=label)
#     ax.axvline(40, c='gray', ls=':')
    
#     ax.set(
#         xlabel=f'Altitude ({altitude.unit})',
#         ylabel='Abundance',
#         title=f'Atmospheric Abundances on {data.meta["datetime"]}, {data.meta["lat"]}, {data.meta["lon"]}',
#         yscale='log',
#         xlim=(altitude.min().value, altitude.max().value),
#         ylim=(1e-5, 1)
#     )
#     ax.legend()

#     return ax


# def plot_abundances_stackplot(data: QTable, ax: plt.Axes = None) -> plt.Axes:
    
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    
#     density_cols = ['Oden', 'N2den', 'O2den', 'Heden', 'Arden', 'Hden', 'Nden']
#     abund_cols = [c.replace('den', '') + 'abund' for c in density_cols]
#     labels = [c.replace('abund', '') for c in abund_cols]
#     y = np.array([list(r.values()) for r in data[abund_cols]]).T * 100
#     altitude = data['Heit']

#     ax.stackplot(altitude.value, y, labels=labels)

#     ax.set(
#         xlabel=f'Altitude ({altitude.unit})',
#         ylabel='Abundance',
#         title='Atmospheric Abundances on Dec. 01, 2023, McMurdo Station, Antarctica',
#         yscale='log',
#         xlim=(altitude.min().value, altitude.max().value),
#         ylim=(75, 100)
#     )
#     ax.legend()

#     return ax


# def main():

#     parser = argparse.ArgumentParser(
#         description='Plots the atmospheric composition for the given pickle file. \
#             See generate_atmospheric_lookup_table.py for how to generate the pickle file. \
#             Plots are saved in the same directory as the pickle file.',
#         epilog='Example of use: python plot_atmospheric_composition.py -i ./mcmurdo-model-data/atmosphere-composition.pkl'
#     )
#     parser.add_argument('-i', type=str, help='pickle file containing atmopsheric data')
#     arg = parser.parse_args()
#     pickle_file = arg.i
#     dir_name = os.path.dirname(pickle_file)
#     data = read_data(pickle_file)

#     # plot_densities(data)

#     plot_abundances(data)
#     plt.savefig(f'{dir_name}/abundances.png', dpi=200)

#     plot_abundances_stackplot(data)
#     plt.savefig(f'{dir_name}/abundances-stacked.png', dpi=200)


# if __name__ == '__main__':
#     main()