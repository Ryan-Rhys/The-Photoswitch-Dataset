# The Photoswitch Dataset

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides benchmarked property prediction results on a curated dataset of 405 photoswitch molecules. 

<p align="center">
  <img src="photoswitch_logo.png" width="500" title="logo">
</p>

# Installation

We recommend using a virtual environment to run the property prediction scripts.

```
conda create -n photoswitch python=3.7

conda install -c conda-forge rdkit

conda install umap-learn seaborn xlrd matplotlib ipython pytest pytorch scikit-learn

pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow

pip install dgllife

pip install dgl
```

# What we provide

The dataset includes molecular properties for 405 photoswitch molecules. 
All molecular structures are denoted according to the simplified molecular input line entry system (SMILES). We collate
the following properties for the molecules:

**Rate of thermal isomerisation** (units = s^-1): This is a measure of the thermal stability of the least stable 
isomer (Z isomer for non-cyclic azophotoswitches and E isomer for cyclic azophotoswitches). Measurements are carried out 
in solution with the compounds dissolved in the stated solvents.

**Photo Stationary State** (units = % of stated isomer): Upon continuous irradiation of an azophotoswitch a 
steady state distribution of the E and Z isomers is achieved. Measurements are carried out in solution with the 
compounds dissolved in the ‘irradiation solvents’.

**pi-pi-star/n-pi-star wavelength** (units = nanometers): The wavelength at which the pi-pi*/n-pi* electronic transition 
has a maxima for the stated isomer. Measurements are carried out in solution with the compounds dissolved in the 
‘irradiation solvents’.

**DFT-computed pi-pi-star/n-pi-star wavelengths** (units = nanometers): DFT-computed wavelengths at which the
pi-pi*/n-pi* electronic transition has a maxima for the stated isomer.

**Extinction coefficient**: The molar extinction coefficient.

**Wiberg Index**: A measure of the bond order of the N=N bond in an azophotoswitch. Bond order is a measure of the 
‘strength’ of said chemical bond. This value is computed theoretically.

**Irradiation wavelength**: The specific wavelength of light used to irradiate samples from E-Z or Z-E such that 
a photo stationary state is obtained. Measurements are carried out in solution with the compounds dissolved in the 
‘irradiation solvents’.

