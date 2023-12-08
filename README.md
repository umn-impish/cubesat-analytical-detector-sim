# Analytical detector model (for IMPRESS mostly; easily extendable)

## Installation
No official way (super under-development package).
Just add the repo folder to your PYTHONPATH inside your `bashrc` or `zshrc` or what have you:
```
    export PYTHONPATH="/path/to/cubesat-analytical-detector-sim:$PYTHONPATH"
```

## Data sources
### attenuation data from
- https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
- https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

## How to add new elemens to the simulation
1. Get space-delimited data from the [XCOM database](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html)
2. Run it through `clean_format_xcom.py` to format it in the correct way
3. Load the produced `.tab` file in using the `AttenuationData` class
4. Instantiate a `Material` givinig it the `AttenuationData` class.
Then you can put the material into a `DetectorStack` or just compute attenuations on it directly.
