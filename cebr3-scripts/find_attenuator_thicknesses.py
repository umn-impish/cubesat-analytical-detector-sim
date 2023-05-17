import numpy as np
import os
import pickle

import adetsim.hafx_src.HafxMaterialProperties as hmp
from adetsim.hafx_src.HafxMaterialProperties import SINGLE_DET_AREA
from adetsim.sim_src.Material import Material
from adetsim.sim_src.FlareSpectrum import FlareSpectrum
from adetsim.hafx_src.HafxStack import HafxStack
from adetsim.sim_src.AttenuationData import AttenuationData

MODEL_ENERGY_EDGES = np.arange(10, 300, 0.1)

def count_edge(cts: float, target: float, step_sgn: float):
    '''
    when we want to change something in the thickness-finding loop.
        with positive increment, cts < target, i.e. too much attenuation
        with negative increment, cts > target, i.e. attenuator is too thin
    '''
    if step_sgn > 0: return cts < target
    elif step_sgn < 0: return cts > target
    else: raise ValueError("step is indistinguishible from zero")


def appr_count_step(det_stack, flare, target_cps):
    '''
    start with thickness that's sure to attenuate the flare
    zigzag around target count rate until we're close enough for gov't work (below and within 1%)
    '''
    eng = flare.energy_edges
    step = -det_stack.att_thick / 2
    divs = 0
    TOL = 0.01
    MAX_DIVS = 32

    while divs < MAX_DIVS and det_stack.att_thick > (-1e-6):
        print(f"{flare.goes_class}: {det_stack.att_thick:.4e} cm")
        res = det_stack.generate_detector_response_to(flare, disperse_energy=True)
        counts_per_kev = res @ flare.flare * SINGLE_DET_AREA
        cur_counts = np.sum(np.diff(eng) * counts_per_kev)

        print("Counts: ", cur_counts)
        if count_edge(cur_counts, target_cps, step):
            print("Found the count edge.\n", f"Counts: {cur_counts}, thickness: {det_stack.att_thick:.4e} cm")
            step /= -2
            divs += 1

        det_stack.att_thick += step
        delta = 1 - cur_counts/target_cps
        if abs(delta) < TOL and delta > 0:
            print('found good one!')
            break

    if divs == MAX_DIVS:
        print("** hit max number of step divisions.")
    if det_stack.att_thick < 0:
        print("** zero attenuator window thickness! uh oh")
    # go back a step and cut off precision at 1e-6 cm
    clean_thick = det_stack.att_thick - step
    if clean_thick < 1e-6: clean_thick = 0
    det_stack.att_thick = clean_thick


def find_appropriate_counts(class_thick, target_cps, material_key):
    ''' optimize attenuator window for target_cps given various GOES flare sizes
        new 2023: change the filter material
    '''
    os.makedirs('responses-areas', exist_ok=True)
    for gc, thick in class_thick.items():
        fs = FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc,
            energy_edges=MODEL_ENERGY_EDGES
        )

        stack = HafxStack(enable_scintillator=True, att_thick=thick)
        mat = Material(
            stack.materials[0].diameter, thick, hmp.DENSITIES[material_key],
            AttenuationData.from_nist_file(hmp.ATTEN_FILES[material_key]),
            material_key
        )
        # hot-swap the attenator
        stack.materials[0] = mat

        # populates matrices of detector_stack
        appr_count_step(stack, fs, target_cps)

        thick = stack.materials[0].thickness
        out_fn = f'responses-areas/optimized-{gc}-{thick:.3e}cm-hafx-{material_key}.pkl'
        with open(out_fn, 'wb') as f:
            pickle.dump(stack, f)
        print(f"saved {out_fn}.")


if __name__ == '__main__':
    # need to be greater than necessary (loop starts by decr. thickness)
    class_thickness = {
        'C1': 0.5,
        'M1': 0.5,
        'M5': 0.5,
        'X1': 0.5,
    }

    '''
    here basically the tau variable he describes is equivalent to
    the hold_off_time set in the Bridgeport detectors.
    the shortest the hold_off_time can be is the integration_time.
    the integration_time that MSU has been using is 48 clock cycles (1.2 us).
    so the minimum dead time (what we want to be using for calculations here) is 1.2 us.
    '''

    bridgeport_measured_dead_time = 1.2e-6
    target_pileup_prob = 0.05

    # see Knoll ch 17 eq 17.7
    target_cps = -np.log(1 - target_pileup_prob) / bridgeport_measured_dead_time
    print(f'goal counts/sec: {target_cps:.2f}')
    find_appropriate_counts(class_thickness, target_cps, 'Al')

