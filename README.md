
# How Physically Robust Are Machine Learning Force Fields For Water?

## Overview

This project investigates the physical robustness of machine learning force fields (MLFFs) for modelling water during molecular dynamics simulations. The study developed a series of machine learning force fields with simple architectures to identify potential failure modes for water, and identify to what extent can they reliably reproduce the underlying physics of water. Model performace was evaluated using both conventional machine learning metrics, trajectory analysis, and physically informed tests.

## Requirements

* Python
* PyTorch
* ASE
* TBLite
* CP2K
* NumPy
* pandas 
* Matplotlib
* PyTorch Geometric
* GMD MACE via https://github.com/Nourollah/GMD-26

## Models Developed 
* Multi-Layer Perceptron
* Convolutional Network 
* Multi-head Attention 
* Combined Black Box 

## Dataset
* Water
* 2500 frames
* 80/20 training/test split
* Reference method: GFN2-xTB

## Directory Structure

Trjectories and models saved as .pt files are too large to directly submit to the submission point.

* model_training.py builds and trains models, as well as runs molecular dynamics simulations.
* equilibration.py contains equilibration run to generate training data, generating water_dataset_64.extxyz
* /analysis contains trajectory files, csvs and result plots from MLFF trajectories, and the all_tests.py script to run all analysis on trajectories.


## Running the Code

### 1. Train the models and run molecular dynamics to acquire trajectories

The training script builds the multi-layer perceptron, convolutional, multi-head attention, black box and commercially available MACE networks, trains them, and runs molecular dynamics simulations to obtain their trajectories.

Run:

```bash
python model_training.py
```


### 2. Analyse the trajectories

Run:

```bash
python all_tests.py
```

## Analysis

The project evaluates:

* Energy and force MAE
* Energy drift
* Energy and force conservation
* Rotational invariance/equivariance
* Radial distribution functions
* Hydrogen bonding behaviour

## Reproducibility

The scripts should be run in the following order:

**Training → MD simulations → trajectory analysis → results/figures**

## Limitations
* Simulation length and water box size
* Complexity of ML models 
* Additional, more rigorous analysis of trajectories could be performed with more time and computational resources


## Author

Summer Zhao, supervised by Dr Cameron Beevers
MSc Scientific Computing with Data Science, University of Bristol
