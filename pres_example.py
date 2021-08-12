import numpy as np
import matplotlib.pyplot as plt
import os

from sim_src.AttenuationData import AttenuationType
from sim_src.FlareSpectrum import FlareSpectrum
from HafxStack import HAFX_MATERIAL_ORDER
from HafxSimulationContainer import HafxSimulationContainer


def load_example(remake: bool) -> HafxSimulationContainer:
    out_dir = HafxSimulationContainer.DEFAULT_SAVE_DIR
    base_fn = 'box_example'
    if not remake:
        try:
            ex_f = next(fn for fn in os.listdir(out_dir) if base_fn in fn)
            print('loading')
            return HafxSimulationContainer.from_saved_file(os.path.join(out_dir, ex_f))
        except StopIteration:
            pass

    print("creating from scratch this time")
    num_energies = 1000
    ex_spectrum = np.zeros(num_energies)
    start_idx = 10
    end_idx = 100
    ex_spectrum[start_idx:end_idx] = 1e6      # photons/sec i guess
    energies = np.linspace(1, 300, num=num_energies)
    ex_fs = FlareSpectrum('', energies, ex_spectrum, np.zeros(num_energies))

    example_thick = 0.1        # cm
    con = HafxSimulationContainer(aluminum_thickness=example_thick, flare_spectrum=ex_fs, save_minimal=False)
    con.simulate()
    con.save_to_file(prefix=base_fn)
    return con

#container = load_example(True)
target_dir = 'optimized-9-aug-2021'
opt_fn = next(fn for fn in os.listdir(target_dir) if 'M5' in fn)
container = HafxSimulationContainer.from_saved_file(os.path.join(target_dir, opt_fn))

# pull out the stuff we need from the container
ds = container.detector_stack
fs = container.flare_spectrum
att_spectra = list()
att_spectra.append(fs.flare)
for mat_obj in ds.materials:
    att_mat = mat_obj.generate_overall_response_matrix_given(fs, AttenuationType.ALL)
    try:
        rec_spectrum = att_spectra[-1]
    except IndexError:
        print('index error')
        rec_spectrum = fs.flare
    att_spectra.append(np.matmul(att_mat, rec_spectrum))

# scintillator is different bc it doesn't "attenuate" the signal--it "makes" it!
scint_resp = ds.generate_scintillator_response(fs, AttenuationType.ALL)
att_spectra.append(np.matmul(scint_resp, att_spectra[-1]))

energy_resolution_mtx = ds.photon_detector.generate_energy_resolution_given(fs)
att_spectra.append(np.matmul(energy_resolution_mtx, att_spectra[-1]))

material_ord = ['Original'] + HAFX_MATERIAL_ORDER + ['Energy dispersion']
fig, ax = plt.subplots()
ax.set_ylim(1e-4, 1e9)
# ax.set_xlim(20, 150)
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Example spectrum (counts / keV / cm${}^2$ / s)')
ax.set_yscale('log')
ax.set_xscale('log')
fig.set_size_inches(8, 6)
i = 0
for (spectrum, mat_name) in zip(att_spectra, material_ord):
    ax.set_title('M5 example solar flare')
    ax.plot(fs.energies, spectrum + 1e-8, label=f"{'After ' if i > 0 else ''}{mat_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/progressive_attenuation{i}.png")
    i += 1
