import numpy as np

from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from adetsim.hafx_src.HafxMaterialProperties import HAFX_DEAD_TIME, SINGLE_DET_AREA
from adetsim.sim_src.FlareSpectrum import FlareSpectrum

HSC = HafxSimulationContainer
mine, maxe, de = HSC.MIN_ENG, HSC.MAX_ENG, HSC.DE
MODEL_ENERGY_EDGES = np.arange(
    HafxSimulationContainer.MIN_ENG,
    HafxSimulationContainer.MAX_ENG + HafxSimulationContainer.DE,
    step=HafxSimulationContainer.DE
)

def battaglia_iter(goes_classes: str):
    for gc in goes_classes:
        yield FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc,
            energy_edges=MODEL_ENERGY_EDGES
        )


def count_edge(cts, target, step_sgn):
    '''
    when we want to change something in the thickness-finding loop.
        with positive increment, cts < target, i.e. too much attenuation
        with negative increment, cts > target, i.e. attenuator is too thin
    '''
    if step_sgn > 0: return cts < target
    elif step_sgn < 0: return cts > target
    else: raise ValueError("step is indistinguishible from zero")


def appr_count_step(sim_con, target_cps):
    '''
    start with thickness that's sure to attenuate the flare
    zigzag around target count rate until we're close enough for gov't work (below and within 5%)
    '''
    eng = sim_con.flare_spectrum.energy_edges
    step = -sim_con.al_thick / 2
    divs = 0
    TOL = 0.01
    MAX_DIVS = 32

    while divs < MAX_DIVS and sim_con.al_thick > (-1e-6):
        print(f"{sim_con.flare_spectrum.goes_class}: {sim_con.al_thick:.4e} cm")
        sim_con.simulate()
        res = sim_con.matrices[sim_con.KDISPERSED_RESPONSE]
        counts_per_kev = res @ sim_con.flare_spectrum.flare * SINGLE_DET_AREA
        cur_counts = np.sum(np.diff(eng) * counts_per_kev)

        print("Counts: ", cur_counts)
        if count_edge(cur_counts, target_cps, step):
            print("Found the count edge.\n", f"Counts: {cur_counts}, thickness: {sim_con.al_thick:.4e} cm")
            step /= -2
            divs += 1

        sim_con.al_thick += step
        delta = 1 - cur_counts/target_cps
        if abs(delta) < TOL and delta > 0:
            break

    if divs == MAX_DIVS:
        print("** hit max number of step divisions.")
    if sim_con.al_thick < 0:
        print("** zero attenuator window thickness! uh oh")
    # go back a step and cut off precision at 1e-6 cm
    clean_thick = sim_con.al_thick - step
    if clean_thick < 1e-6: clean_thick = 0
    sim_con.al_thick = clean_thick


def find_appropriate_counts(class_thick, target_cps):
    ''' optimize attenuator window for target_cps given various GOES flare sizes '''
    for gc, thick in class_thick.items():
        fs = FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc,
            energy_edges=MODEL_ENERGY_EDGES
        )
        sim_container = HafxSimulationContainer(
            aluminum_thickness=thick,
            flare_spectrum=fs)
        # populates matrices of detector_stack
        appr_count_step(sim_container, target_cps)
        sim_container.save_to_file(prefix='optimized')
        print(f"Saved {sim_container.gen_file_name('optimized')}.")


if __name__ == '__main__':
    # need to be greater than necessary (loop starts by decr. thickness)
    class_thickness = {
            # 'B5': 1e-2,
            # 'C5': 0.1,
            # 'M5': 0.1,
            # 'X1': 0.1
            # 'C5': 0.1,
            'C1': 1e-4,
            'M1': 80e-4,
            'M5': 250e-4,
            'X1': 350e-4
        }

    target_cps = -np.log(0.95) / HAFX_DEAD_TIME
    find_appropriate_counts(class_thickness, target_cps)
