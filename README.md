# ![tomohowk logo](./doc/logo_cropped.png) Tomohowk
## Introduction

Tomohowk is a toolbox for continuous variable quantum homodyne tomography.
At its heart is the efficient and automatic reconstruction of Wigner functions from quadrature data.
Additionally, it provides facilities for common preprocessing tasks, like the conversion of photocurrent voltages into quadratures and the detection of the phase from steps of the piezo stage, as well as some postprocessing support for visualization and calculation of expectation values.

Tomohowk is written in Python and can make use of multiple cores or CUDA enabled graphics cards to speed up its computations.
All data is handled in the [HDF5 format](https://www.hdfgroup.org/HDF5/).

Details about the usage can be found in [the manual](./doc/manual.pdf).

## License
Tomohowk is licensed under the MIT License. [View the license file.](./LICENSE)
