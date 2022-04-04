; example parameters
; 1.0, 300.0, 0.1, 0.20340744608778885, 19.563430168635527, 1.0
function f_vth_bridge, start_energy, end_energy, de, emission_measure, plasma_temperature, relative_abundance
	thermal_params = [emission_measure, plasma_temperature, relative_abundance]

    ; f_vth works using histogram bins. so weird.
    ; shift back one half-step so the mean bin energy corresponds to a user-requested number
    start_energy -= de
    end_energy += de
	ary_sz = ulong((end_energy - start_energy) / de)
	eng_array = findgen(ary_sz, start=start_energy, increment=de)

    ; we pass 2xN histogram bin edges to f_vth, not just a 1D array.
    ; poorly documented required thing!
    eng_edges = get_edges(eng_array, /edges_2)
	spectrum = f_vth(eng_edges, thermal_params, /mewe)
	return, spectrum
END
