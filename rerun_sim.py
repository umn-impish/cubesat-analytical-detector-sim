from HafxSimulationContainer import HafxSimulationContainer
import os

'''
If simulation logic is updated, use this script to re-run the given simulations using new logic.
'''

dir_of_interest = 'optimized-10-aug-2021'
prefix = 'optimized'
files = os.listdir(dir_of_interest)

for fn in (os.path.join(dir_of_interest, ff) for ff in files):
    print("Working on", fn)
    con = HafxSimulationContainer.from_saved_file(fn)
    con.simulate()
    # con.save_to_file(out_dir=dir_of_interest, prefix=prefix)
