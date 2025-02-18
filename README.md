[![Build Status](https://travis-ci.org/OpenDA-Association/OpenDA.svg?branch=master)](https://travis-ci.org/OpenDA-Association/OpenDA)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e78fb37c00fa4d689eb3a40d4b786d94)](https://www.codacy.com/app/OpenDA_Association/OpenDA?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OpenDA-Association/OpenDA&amp;utm_campaign=Badge_Grade)

# OpenDA

OpenDA is an open interface standard for (and free implementation of) a set of tools to quickly implement data-assimilation and calibration for arbitrary numerical models. OpenDA wants to stimulate the use of data-assimilation and calibration by lowering the implementation costs and enhancing the exchange of software among researchers and end-users.
A model that conforms to the OpenDA standard can use all the tools that are available in OpenDA. This allows experimentation with data-assimilation/calibration methods without the need for extensive programming. Reversely, developers of data-assimilation/calibration software that make their implementations compatible with the OpenDA interface will make their new methods usable for all OpenDA users (either for free or on a commercial basis).
OpenDA has been designed for high performance. Hence, even large-scale models can use it. Also, OpenDA allows users to optimize the interaction between their model and the data-assimilation/calibration methods. Hence, data-assimilation with OpenDA can be as efficient as with custom-made implementations of data-assimilation methods.
OpenDA is an Open Source project. Contributions are welcome from anyone wishing to participate in the further development of the OpenDA toolset.

## Features of OpenDA

Data-assimilation methods

- Ensemble KF (EnKF)
- Ensemble SquareRoot KF (EnSR)
- Steady State KF
- Particle Filter
- 3DVar
- DudEnKF (still under research)
- DudEnSR (still under research)

Parameter estimation (calibration) methods:

- Dud
- Sparse Dud
- Simplex
- Powell
- Gridded full search
- Shuffled Comples Evolution (SCE)
- Generalized Likelihood Uncertainty Estimation (GLUE)
- (L)BFGS
- Conjugate Gradient: Fleetjer-Reeves, Polak-Ribiere, Steepest Descent
- Uncertainty Analaysis methods
- GLUE
- DELSA

Language interfaces

- C/C++
- Java
- Fortran77/90

These files are part of the OpenDA software. For more information see our website at
http://www.openda.org

