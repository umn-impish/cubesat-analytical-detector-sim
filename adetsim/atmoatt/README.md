# atmoatt

This project contains code on simulating the atmospheric attenuation of solar X-rays.
It uses the core `adetsim` module for computing the elemental effects, and the [`pymsis`](https://swxtrec.github.io/pymsis/index.html) API to obtain elemental abundances as a function of time of year, location on Earth, and altitude.
See the `projects/grip-piggyback` directory for examples.


## Solar zenith angle

This code accommodates the zenith angle of the Sun when simulating the propagation of solar X-rays through the atmosphere.
This is done using a simple calculation that can be found [here](https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Homogeneous_atmosphere).
