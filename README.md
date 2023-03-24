# HSolver
The tool for solving quantum state evolution under periodic hamiltonian.

## Getting started

To install the package download the files and run
```commandline
python setup.py install
```
or without downloading
```commandline
pip install git+https://github.com/bbantysh/hsolver.git
```

## Documentation

For documentation see docstrings and [simple examples](examples/).

## Basic features

* Flexible interface for constructing Hamiltonian.
* Fast computation using sparse matrices.
* Fast computation of periodic time intervals using [Floquet theory](https://en.wikipedia.org/wiki/Floquet_theory).

## Recommendations

For faster computation it is sometimes recommended to approximate hamiltonian envelopes with a specific form.
For example,
* Consider using interaction picture to avoid large frequencies.
* Given two hamiltonian terms with periods T1 and T2 acting simultaneously, consider adjusting periods to make the rational approximation T1/T2 = p/q to have small p and q. Fox example T2 = 2 * T1 works mush faster than T2 = 2.0001 * T1. 
* Consider approximating slowly varying envelopes with constant/periodic ones (maybe with some non-zero but small front width). 


## License
All code found in this repository is licensed under GPL v3.
