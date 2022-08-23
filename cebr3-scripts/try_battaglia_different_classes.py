from adetsim.sim_src.FlareSpectrum import BattagliaParameters, goes_class_lookup

test_gc = ['C2.1', 'C4.0']

for gc in test_gc:
    flux = goes_class_lookup(gc)
    bp = BattagliaParameters(flux)
    print(gc, '->', bp)
