import numpy as np
import os
import subprocess
import sys
'''
    some helper functions to connect Python to specific parts of sswidl.
    requires respective "bridge" script in the same directory, for example f_vth_bridge.pro.
    each function returns... well, the expected output. for example, for f_vth, it returns a numpy.ndarray.

    this is a hack to get the interface working but the official sswidl bridge will hopefully replace this script soon.
'''

SSW_IDL_LOCATION = "/usr/local/ssw/gen/setup/ssw_idl"
CMD_DONE_IDENTIFIER = '*-*' * 10

def power_law_with_pivot(eng_ary, reference_flux, spectral_index, e_pivot):
    # same code as f_1pow
    # just a power law at some reference energy
    return reference_flux * ((e_pivot / eng_ary) ** (spectral_index))


def f_vth_bridge(eng_start, eng_end, de, emission_measure, plasma_temperature, relative_abundance):
    '''
    NB: energies are in keV
        *** emission measure is in (1e49 cm-3) ***
        *** plasma_temperature is in keV ***

    the wonky units are to match what IDL has. :(
    '''
    args = [eng_start, eng_end, de, emission_measure, plasma_temperature, relative_abundance]
    dirty = run_sswidl_script("f_vth_bridge", *args)
    return clean_table_output(dirty)


def run_sswidl_script(script_name, *args, debug=False):
    '''
    The idea here is th compile and run the desired function, and its output gets delimited by CMD_DONE_IDENTIFIER.
    Then we can just split off the part of SSWIDL's ramblings that we care about, our function output.
    Then we just return it and let the user decide what to do with the rest of it.
    '''
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    str_args = [str(a) for a in args]
    idl_cmds_ary = [
        '.compile {sp}'.format(sp=script_path),
        'print, "{cd}"'.format(cd=CMD_DONE_IDENTIFIER),
        '{sn}({argz})'.format(
            sn=script_name,
            argz=','.join(str_args)),
        'print, "{cd}"'.format(cd=CMD_DONE_IDENTIFIER),
        'exit'
    ]
    idl_cmds = '\n'.join(idl_cmds_ary)
    idl_cmds = bytes(idl_cmds, 'utf-8')

    idl_proc = subprocess.Popen(SSW_IDL_LOCATION, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_result = idl_proc.communicate(idl_cmds)
    # isolate stdout
    out = process_result[0].decode('utf-8')
    if debug:
        loud_print(out)
    out = out.split(os.linesep)

    # slice out the output
    # list slices are [inclusive:exclusive)
    estr = "Nothing returned from IDL. check that your script isn't broken/missing. it has to be in the same folder as the bridge file"
    try:
        data_start_idx = out.index(CMD_DONE_IDENTIFIER) + 1
        data_end_idx = out.index(CMD_DONE_IDENTIFIER, data_start_idx)
        out = out[data_start_idx:data_end_idx]
        if len(out) == 0:
            raise ValueError(estr)
        return out
    except ValueError as e:
        # if we made the value error just re-raise it
        if estr == e.args[0]:
            raise
        loud_print("The IDL script likely didn't run. Make sure you're launching from tcsh (shudders)", file=sys.stderr)
        input("Press any key to continue")
        raise


def loud_print(*args, **kwargs):
    print('*' * 30, **kwargs)
    print(*args, **kwargs)
    print('*' * 30, **kwargs)


def clean_table_output(res):
    return np.array([float(y) for x in res for y in x.strip().split()], dtype=np.float64)


if __name__ == '__main__':
    from FlareSpectrum import BattagliaParameters
    vth_name = "f_vth_bridge"
    pow_name = "f_1pow_bridge"
    K_B = 8.627e-8          # keV/K
    ENG_START = 1.0         # keV
    ENG_END = 300.0         # keV
    DE = 0.1                # keV
    goes_intensity = 1e-5   # W / m2

    bp = BattagliaParameters(goes_intensity)
    energy_vec = np.arange(ENG_START, ENG_END + DE, DE, dtype=np.float64)
    print(energy_vec)

    # just a power law
    nonthermal = power_law_with_pivot(energy_vec, bp.flux_35kev, bp.spectral_index, 35.0)
    pt, em = bp.gen_vth_params()
    #           eng_start, eng_end, de,  emission_measure, plasma_temp, relative_abundance
    vth_args = [ENG_START, ENG_END, 0.1, em, pt, 1.0]
    print(vth_args)
    vth_res = run_sswidl_script(vth_name, *vth_args)
    thermal = clean_table_output(vth_res)

    total_spectrum = nonthermal + thermal
    np.savetxt('spectrum.tab', np.transpose((energy_vec, total_spectrum)), delimiter='\t')
