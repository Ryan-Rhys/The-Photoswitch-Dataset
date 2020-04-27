# The Molecular Photoswitch Dataset

This repository provides a curated dataset of the properties of 98 molecular photoswitch molecules.

# Installation

We recommend using a virtual environment to run the property prediction scripts.

```
conda create -n photoswitch python=3.7

conda install scikit-learn==0.22.1

conda install rdkit==2019.09.2

pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
```

# What we provide

The dataset includes molecular properties for 98 photoswitch molecules. 
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

**Extinction coefficient**: See “D. W. Ball, Field Guide to Spectroscopy, SPIE Press, Bellingham, WA (2006)” 
for an introduction into the Beer-Lambert Law and its applications in photochemistry.

**Wiberg Index**: A measure of the bond order of the N=N bond in an azophotoswitch. Bond order is a measure of the 
‘strength’ of said chemical bond. This value is computed theoretically.

**Irradiation wavelength**: The specific wavelength of light used to irradiate samples from E-Z or Z-E such that 
a photo stationary state is obtained. Measurements are carried out in solution with the compounds dissolved in the 
‘irradiation solvents’.

