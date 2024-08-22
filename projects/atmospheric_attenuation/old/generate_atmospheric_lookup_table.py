# import argparse
# import astropy.units as u
# import numpy as np
# import parse
# import pickle

# from astropy.table import QTable


# def txt_to_csv(txt_file: str) -> str:
#     """
#     Cleans and converts the text file to a CSV file.
#     The input text file should be the raw output from
#     https://kauai.ccmc.gsfc.nasa.gov/instantrun/nrlmsis/
#     """

#     with open(txt_file, 'r') as infile:
#         text = infile.read()
    
#     # Remove first line if it's not data
#     if text.startswith('Since the'):
#         text = text[text.index('\n')+1:]

#     # Remove whitespace and replace with commas while presevering \n
#     text = text.replace('\n', '\\n')
#     text = ','.join(text.split())
#     text = text.replace('\\n', '\n')

#     # Make corrections to certain items
#     text = text.replace('Heden(cm-3)Arden(cm-3)', 'Heden(cm-3),Arden(cm-3)')
#     text = text.replace('gm/cm3', 'g/cm3')

#     csv_file = txt_file.replace('.txt', '.csv')
#     with open(csv_file, 'w') as outfile:
#         outfile.write(text)

#     return csv_file


# def generate_table(csv_file: str) -> QTable:

#     pickle_file = csv_file.replace('.csv', '.pkl')
#     COLUMN_LABEL_FORMAT = '{name}({unit})'
#     data = QTable.read(csv_file, delimiter=',')
#     for column_name in list(data.columns):
#         p = parse.parse(COLUMN_LABEL_FORMAT, column_name)
#         if p is not None:
#             new_name, unit = p['name'], p['unit']
#             data.rename_column(column_name, new_name)
#             data[new_name].unit = u.Unit(unit)

#     compute_abundances(data)

#     with open(pickle_file, 'wb') as outfile:
#         pickle.dump(data, outfile)

#     print('Wrote table to', pickle_file)

#     return data


# def compute_abundances(data: QTable):
#     """
#     Computes atmospheric abundances from the densities of each element and molecule.
#     """

#     density_cols = ['Oden', 'N2den', 'O2den', 'Heden', 'Arden', 'Hden', 'Nden']
#     indices = [list(data.columns).index(c) + 1 for c in density_cols]
#     abund_cols = [c.replace('den', '') + 'abund' for c in density_cols]

#     abundances = np.array([data[c].value for c in density_cols])
#     totals = np.sum(abundances, axis=0)
#     abundances = abundances / totals
#     data.add_columns(list(abundances), indexes=indices, names=abund_cols)


# def main():

#     parser = argparse.ArgumentParser(
#         description='Converts the model data (txt file) from \
#             https://kauai.ccmc.gsfc.nasa.gov/instantrun/nrlmsis/ into a csv file \
#             and an astropy QTable (save as pkl) with computed abundances. Simply save the model \
#             data from the website into a txt file and pass that file as input \
#             into this script.',
#         epilog='Example of use: python generate_atmospheric_lookup_table.py -i model-data/atmospheric-composition-vs-altitude-full-height-step5.txt'
#     )
#     parser.add_argument('-i', type=str, help='text file containing atmospheric model data')
#     arg = parser.parse_args()
#     txt_file = arg.i

#     csv_file = txt_to_csv(txt_file)
#     generate_table(csv_file)


# if __name__ == '__main__':
#     main()