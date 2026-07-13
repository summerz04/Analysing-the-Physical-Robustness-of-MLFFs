# analysis of water trajectories

import os 
import numpy as np 

# H bond network analysis 
from CRISP.data_analysis.h_bond import hydrogen_bonds, indices 

# list of trajectories from ase
files = ['xtb_test.traj']

# parameters for hydrogen bond analysis 
frame_skip = 50 # to prevent crashing 
angle_cutoff = 120

print('Beginning H bond analysis...')

for file in files:
    print(f'Reading {file}')
    
    hydrogen_bonds(
    traj_path=file,
    frame_skip=frame_skip,
    acceptor_atoms=['O'],
    mic=True,
    output_dir='./H_bond_Data',
    plot_heatmap=True
    )
