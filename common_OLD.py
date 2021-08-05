from datetime import datetime
import logging
import numpy as np
import os

import sim_src.impress_constants as ic
from sim_src.AttenuationData import AttenuationData
from HafxStack import HafxStack
from sim_src.FlareSpectrum import FlareSpectrum
from sim_src.Material import Material
from sim_src.PhotonDetector import Sipm3000


def OLD_setup_structure(ident: str) -> None:
    for p in (ic.DATA_DIR, ic.LOGS_DIR):
        if not os.path.exists(p): os.mkdir(p)
    dt = datetime.now()
    lfn = os.path.join(ic.LOGS_DIR, dt.strftime(f"%Y.%m.%d-%H:%M:%S-{ident}.log"))
    logging.basicConfig(filename=lfn)


def print_log(*args):
    print(*args)
    logging.info(*args)


def OLD_load_optimized_hafx(data_dir, goes_classes):
    files = os.listdir(data_dir)
    ideal = dict()
    for f in files:
        sz = f.split('_')[0]
        if sz in goes_classes:
            dat = np.load(os.path.join(data_dir, f))
            k = optimized_key(sz)
            ideal[k] = dat
    return ideal


# XXX: idea, Thresholds class that has energy limits and threshold counts
def OLD_compute_optimized_quantities(ideal: dict, sim_goes_classes: tuple, thresh_counts: int) -> dict:
    loaded = dict()
    korig, katt, keffa = 'orig', 'att', 'effa'
    loaded[katt] = { sz : {} for sz in sim_goes_classes }
    loaded[keffa] = { sz : {} for sz in sim_goes_classes }
    loaded[korig] = dict()
    for fsz in sim_goes_classes:
        print(f"loading {fsz}")
        fs = FlareSpectrum.make_with_battaglia_scaling(fsz, ic.E_MIN, ic.E_MAX, ic.DE)
        loaded[korig][fsz] = fs.flare
        restrict = np.logical_and(fs.energies >= ic.E_TH_MIN, fs.energies <= ic.E_TH_MAX)

        for kopt in ideal.keys():
            att_spec = np.matmul(ideal[kopt][ic.RESP_KEY], fs.flare)
            loaded[katt][fsz][kopt] = att_spec

            loaded[keffa][fsz][kopt] = ideal[kopt][ic.EFFA_KEY]
            if thresh_counts > 0:
                cps = np.trapz(att_spec[restrict] * ic.SINGLE_DET_AREA, x=fs.energies[restrict])
                # reject if detector cant keep up AND we want this behavior
                if cps > thresh_counts:
                    loaded[keffa][fsz][kopt] = np.zeros(fs.energies.size)

    return loaded
