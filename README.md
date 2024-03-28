# `doom`: **D**iffuse b**OO**sted dark **M**atter yielded from supernova neutrinos in early universe


## Introduction

`doom` is a package for evaluating the diffuse signature of dark matter boosted in the early Universe due to supernova neutrinos based on `arXiv:`.

### Installation

To install, excute following command on the terminal:

    $ pip install doom

and everything should be processed on-the-fly.

### Dependencies

`doom` requires these external packages

- `numpy` >= 1.20.0
- `scipy` >= 1.10.0
- `vegas` >= 6.0.1

where `vegas` is a package for evaluating multidimensional integrals using adaptive Monte Carlo vegas algorithm, see its homepage: [https://pypi.org/project/vegas/](https://pypi.org/project/vegas/). Additional package, e.g. `gvar`, maybe required by `vegas` during the installation.
The versions of the external packages are not strict, but we recommend to update to latest ones to avoid incompatibility. 


## Usage

We briefly summarize the usage in this section and a comprehensive tutorial can be found in the jupyter notebook `tutorial/tutorial.ipynb`.

To import, do

    >>> import dbdm

in the python terminal. All documented module functions can be called like `dbdm.`*`funcname`*. 
