# Analytical detector model (for IMPRESS mostly; easily extendable)

## Installation
No official way (super under-development package).
Just add the `adetsim` folder to your PYTHONPATH inside your `bashrc` or `zshrc` or what have you:
```
    export PYTHONPATH="path/to/diagnostic_detector_sim:$PYTHONPATH"
```

## Data sources
### attenuation data from
- https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
- https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html


### information on cerium bromide crystal used in HaFX may be found at
- https://www.berkeleynucleonics.com/cerium-bromide


### characteristic X-ray data
- photoionization cross sections for K, L, M shells from Scofield, J.H. (1973), Theoretical Photoionization Cross Sections from 1 to 1500 keV, Lawrence Livermore Laboratory Report UCRL-51326.
- relative line intensities scraped from LBL X-ray Data Handbook 
