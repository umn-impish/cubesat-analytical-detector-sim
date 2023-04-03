import numpy as np

from adetsim.sim_src import AttenuationData as ad
from adetsim.sim_src import DetectorStack as ds
from adetsim.sim_src import FlareSpectrum as fs
from adetsim.sim_src import Material as mat
from adetsim.hafx_src import FixedEnergyResolutionDetector as ferd

ATT_DIR = 'att-data/cleaned'
att_concat = lambda s: f'{ATT_DIR}/{s}'
MAT_FNS = {
    'teflon': att_concat('teflon.txt'),
    'pla': att_concat('pla.txt'),
    'gagg': att_concat('gagg.txt'),
    'lyso': att_concat('lyso.txt')
}

MAT_DAT = {
    k: ad.AttenuationData.from_nist_file(v)
    for (k, v) in MAT_FNS.items()
}

MAT_DENS = {
    'teflon': 2.2, # g/cm3
    'pla': 1.24,   # g/cm3
    'lyso': 7.1,   # g/cm3
    'gagg': 6.63   # g/cm3

}

class GripsStack(ds.DetectorStack):
    DIAMETER = 2 # cm
    def __init__(self, stack_mats: dict[str, float]):
        det_mats = ('lyso', 'gagg')
        self.detector = None
        mats = []
        for (n, t) in stack_mats.items():
            kw = dict(
                diameter=GripsStack.DIAMETER,
                attenuation_thickness=t,
                mass_density=MAT_DENS[n],
                attenuation_data=MAT_DAT[n],
                name=n
            )
            if n not in det_mats:
                m = mat.Material(**kw)
                mats.append(m)
                continue
            self.detector = mat.Material(**kw)

        res = 0.4
        super().__init__(mats, ferd.FixedEnergyResolutionDetector(res))

    def generate_detector_response_to(
        self, spectrum: fs.FlareSpectrum, disperse_energy: bool,
        chosen_attenuations: tuple=ad.AttenuationType.ALL
    ) -> np.ndarray:
        prelim_resp = self._generate_material_response_due_to(spectrum, chosen_attenuations)
        abzorbed = np.ones(spectrum.energy_edges.size - 1) -\
            self.detector.generate_overall_response_matrix_given(
                spectrum, [ad.AttenuationType.PHOTOELECTRIC_ABSORPTION])
        return self._dispatch_dispersion(spectrum, abzorbed * prelim_resp, disperse_energy)
