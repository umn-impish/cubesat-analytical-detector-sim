; example parameters
; 1.0, 300.0, 0.1, 0.20340744608778885, 19.563430168635527, 1.0
function f_vth_bridge, start_energy, end_energy, de, emission_measure, plasma_temperature, relative_abundance
	thermal_params = [emission_measure, plasma_temperature, relative_abundance]
	; f_vth excludes last energy. wtf?
	ary_sz = ulong((end_energy - start_energy) / de) + 1
	eng_array = findgen(ary_sz, start=start_energy, increment=de)
	spectrum = f_vth(eng_array, thermal_params)
	return, spectrum
END
