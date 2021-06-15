import numpy as np
import matplotlib.pyplot as plt
import os

import impress_constants as ic

# ['flare_spectrum', 'energies', 'response_matrix', 'eff_area']

'''
    what do we want to do
        1) integrate a spectrum to get a total number of counts
        2) verify effective area in the 11 - 26 keV range
        3) ?
'''

def sort_by_goes(fil):
    ''' group by GOES class '''
    sz, t = fil.split('_')[:2]
    mul = float(sz[1])
    t = float(t)
    return ord(sz[0]) + mul + t


def diagnose_areas(df_list, fig, ax):
    C1_TH = 13.6    # cm2
    M1_TH = 1.0     # cm2
    E_ST = 11.0     # keV
    E_END = 26.0    # keV
    xmin = 1        # keV
    xmax = 300      # keV
    ymin = 0        # cm2
    ymax = 14       # cm2

    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Incident x-ray photon energy (keV)")
    ax.set_ylabel("Single-detector effective area (cm${}^2$)")
    ax.set_title(f"Single-detector effective area, aluminum thickness 10$\mu$m to 190$\mu$m")

    # pick one set of flare sizes bc effective area is the same for all of them.
    # let's do C1 because why not
    focus = [f for f in df_list if 'C1' in f]
    for i, foc in enumerate(focus):
        bname = foc.split('_')[1]
        with np.load(os.path.join(ic.DATA_DIR, foc)) as df:
            effective_area = df[ic.EFFA_KEY]
            energies = df[ic.ENG_KEY]
        ax.plot(energies, effective_area)
        print(f"Done with {bname}")

    ax.axhline(y=C1_TH, linestyle='--', label="C1 minimum", color='k')
    ax.axhline(y=M1_TH, linestyle='--', label="M1 minimum", color='r')
    ax.axvline(x=E_ST, linestyle='--', color='blue', label="11 keV")
    ax.axvline(x=E_END, linestyle='--', color='orange', label="26 keV")
    fig.set_size_inches((8, 6))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ic.FIG_DIR, 'combined.pdf'))


# ['flare_spectrum', 'energies', 'response_matrix', 'eff_area']
def total_counts(df_list, fig, ax, fsz):
    LOW_CUTOFF = 8      # keV
    HIGH_CUTOFF = 100   # keV
    for fn in df_list:
        with open("total-counts.txt", "a") as outf:
            if fsz not in fn: continue
            parts = fn.split('_')
            t = parts[1]
            with np.load(os.path.join(ic.DATA_DIR, fn)) as df:
                energies = df['energies']
                fspec = df['flare_spectrum']
                resp = df['response_matrix']
                eff_ar = df['eff_area']

            # cut off spectrum to only include above 8 keV and below 100 keV
            restrict = np.logical_and(energies > LOW_CUTOFF, energies < HIGH_CUTOFF)
            atten = np.matmul(resp, fspec)
            captured = atten * eff_ar
            total_counts = np.trapz(captured[restrict], x=energies[restrict])
            outs = f"Total counts (8 to 100 keV) for {fsz} at Al = {t} cm thick:\t{total_counts:.3e}"
            print(outs)
            print(outs, file=outf)


def main():
    files = os.listdir(ic.DATA_DIR)
    files.sort(key=sort_by_goes)
    f, a = plt.subplots()
    # diagnose_areas(files, f, a)
    flares = ('C1', 'C5', 'M1', 'M5', 'X1')
    for size in flares:
        total_counts(files, f, a, size)

if __name__ == '__main__': main()
