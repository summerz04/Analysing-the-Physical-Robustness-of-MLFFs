import os
os.environ["OMP_NUM_THREADS"] = "24"

import numpy as np
from ase.io import read, write
from ase import units
from ase.build import molecule
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Load or build
if os.path.exists('eq1.xyz'):
    atoms = read('eq1.xyz')
else:
    # build water box of 64 molecules with pbc 
    atoms = molecule('H2O')
    atoms.set_cell((15, 15, 15))
    atoms.center()
    atoms *= (4, 4, 4)
    atoms.set_pbc(True)
    atoms.positions += np.random.uniform(-0.2, 0.2, atoms.positions.shape)

# set up
from tblite.ase import TBLite
atoms.calc = TBLite(method="GFN2-xTB")

# setting up MD parameters 
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)

# write to extxyz for training 
output_file = 'water_dataset_64.extxyz'
if os.path.exists(output_file):
    os.remove(output_file)

total_steps = 7500  # 7500 / 3 = 2500 frames
save_every = 2
frames = []

# run MD 
print(f"Running {total_steps} steps of GFN2-xTB...")
for step in range(total_steps):
    dyn.run(1)
    
    if step % save_every == 0:
        frame = atoms.copy()
        frame.info['energy'] = atoms.get_potential_energy()
        frame.arrays['forces'] = atoms.get_forces()
        frame.calc = None
        frames.append(frame)
        
    # write to disk every 500 frames 
    if len(frames) >= 50 or step == total_steps - 1:
        write(output_file, frames, append=(step >= 500))
        frames = []
        print(f"Saved {min(step + 1, total_steps)} / {total_steps} steps")

print(f"DONE! File saved to {output_file}")
