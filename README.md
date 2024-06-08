# approximate_neural_lyapunov_control
This repository contains code for approximate neural lyapunov control - a framework that simultaneously learns a controller and a Lyapunov function for systems with unknown non-linear dynamics. A sampling-based falsifier is used to guide the learner towards more robust solutions.

## Requirements

First, create a conda environment with Python 3.8.
```
conda create -n lyapunov python=3.8
```

Activate the environment
```
conda activate lyapunov
```

Install the required packages
```
pip install -r requirements.txt
```

## References
```
@inproceedings{NEURIPS2019_2647c1db,
 author = {Chang, Ya-Chien and Roohi, Nima and Gao, Sicun},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Neural Lyapunov Control},
 url = {https://proceedings.neurips.cc/paper/2019/file/2647c1dba23bc0e0f9cdf75339e120d2-Paper.pdf},
 volume = {32},
 year = {2019}
}
```
