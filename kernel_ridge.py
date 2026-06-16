
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# PRIME ENCODING, labelling elements
prime_assign = {'H': 2, 'O': 7, 'F': 11, 'N': 13, 'C': 17}
log_primes = {el: np.log(p) for el, p in prime_assign.items()}

def generate_molecules(n_h2o=1000, filename='water_set_1000.xyz'):
    """
    Returns [(pos, elements, E_total, F_atom)].
    Also writes filename as XYZ file.
    """
    molecules = []
    positions = []

    # H2O
    for _ in range(n_h2o):
        r_oh = np.random.normal(0.96, 0.1)
        theta = np.deg2rad(np.random.normal(104.5, 5))
        phi = np.random.uniform(0, 2*np.pi)

        O = np.zeros(3)
        H1 = r_oh * np.array([np.sin(theta)*np.cos(phi),
                              np.sin(theta)*np.sin(phi),
                              np.cos(theta)])
        H2 = r_oh * np.array([np.sin(theta)*np.cos(phi+np.pi),
                              np.sin(theta)*np.sin(phi+np.pi),
                              np.cos(theta)])
        pos = np.vstack([O, H1, H2]) + 0.03*np.random.randn(3, 3)
        E_total = lj_potential(pos, 0.1)
        F_atom = lj_forces(pos, 0.1)
        molecules.append((pos, ['O','H','H'], E_total, F_atom))

        # append positions to get list of positions of atom per molecule
        positions.append(pos)
    # Write XYZ
    with open(filename, 'w') as f:
        for i, (pos, els, E, F) in enumerate(molecules):
            f.write(f"{len(els)}\n")
            f.write(f"mol_{i} E={E:.6f}\n")
            for j, el in enumerate(els):
                f.write(f"{el:2s} {pos[j][0]:12.6f} {pos[j][1]:12.6f} {pos[j][2]:12.6f}\n")
    print(f"Generated {len(molecules)} structures → {filename}")

    # return positions list 
    return molecules, positions

# using positions list, calculate distances between atoms in water 
import random 
def interatomic_dist(positions):
    # structure is O H H
    #            x - - -
    #            y - - -
    #            z - - -
    print(f'Number of collections of positions:{len(positions)}')

    print(f'Random example:{random.choice(positions)}') # numpy array 

    dists = []
    for i in positions:
        dist1 = np.linalg.norm(i[0] - i[1])
        dist2 = np.linalg.norm(i[0] - i[2])
    
        dists.append([dist1, dist2])
    dists = np.array(dists)

    print(f'Random pair of interatomic distances: {random.choice(dists)}')
    print(f'Shape of dists array: {dists.shape}')

    return dists

def load_molecules_from_xyz(filename='water_set_1000.xyz'):
    """
    Parse training_set.xyz → (pos, elements, E_total, F_atom).
    Recompute LJ energy; ignore E= from file.
    """
    molecules = []
    positions = []
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        try:
            N = int(line)
        except ValueError:
            i += 1
            continue

        i += 1
        if i >= len(lines):
            break
        # Skip comment line (mol_i E=...)
        i += 1

        pos = np.zeros((N, 3))
        elements = []

        for j in range(N):
            if i >= len(lines):
                raise RuntimeError(f"Premature end of file at molecule with N={N}")
            cols = lines[i].split()
            if len(cols) < 4:
                print(f"Bad line at i={i}: {lines[i]!r}")
                raise ValueError("Line too short; expected at least 4 columns (el x y z)")
            elements.append(cols[0])
            try:
                pos[j] = [float(x) for x in cols[1:4]]
            except ValueError as e:
                print(f"Bad coordinates at line {i}: {lines[i]!r}")
                raise e
            i += 1

        # Recompute LJ energy and forces
        E_total = lj_potential(pos, epsilon=0.1, scale=0.01)
        F_atom = lj_forces(pos, epsilon=0.1)
        molecules.append((pos, elements, E_total, F_atom))
        positions.append(pos)
    print(f"Loaded {len(molecules)} structures from {filename}")
    return molecules, positions


# Scaled Lennard-Jones Potential
def lj_potential(pos, epsilon=0.1, scale=0.1):
    r2 = quadrance(pos) + 1e-10
    r6 = r2**3
    V = 4*epsilon * (r6**(-2) - r6**(-1))
    return np.triu(V, k=1).sum() * scale   # e.g. scale=0.01

def lj_forces(pos, epsilon=0.1, scale=0.1):
    """Numerical gradient of lj_potential → per‑atom forces (N, 3)."""
    eps = 1e-6
    forces = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            pos_p = pos.copy(); pos_p[i, d] += eps
            pos_m = pos.copy(); pos_m[i, d] -= eps
            forces[i, d] = -(lj_potential(pos_p, epsilon, scale)
                           - lj_potential(pos_m, epsilon, scale)) / (2*eps)
    return forces


def quadrance(pos):
    """Squared distances Q_ij = ||r_i - r_j||^2."""
    return squareform(pdist(pos, 'sqeuclidean'))


def molecules_to_tensors(molecules, device):
    """
    molecules: list of (pos, els, E_total, F_atom).

    Returns list of (node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N_atoms)
    for each molecule.
    """
    samples = []
    node_repr = []
    targets_E = []

    energy_per_molecule = []

    for pos, els, E_total, F_atom in molecules:
        N = len(els)

        # 1. Node features
        node_feats = np.array([log_primes[el] for el in els])[:, None]
        node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)

       

        # 4. Positions (requires_grad=True)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device, requires_grad=True)


        samples.append((node_t, pos_t, E_total, N))
        node_repr.append(node_t)
        targets_E.append(E_total)
        energy_per_molecule.append(E_total)

    return samples, node_repr, targets_E, energy_per_molecule

class EnergyDataset(Dataset):
    """
    Each item = (node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N_atoms).
    """
    def __init__(self, samples, energy_per_molecule):
        self.samples = samples
        self.energy_per_m = energy_per_molecule

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Returns:
        node_batch, edge_batch, triplet_batch, pos_batch, E_total_batch, F_batch, N_batch.
    """
    node_batch = [b[0] for b in batch]
    pos_batch = [b[1] for b in batch]
    E_total_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    N_batch = [b[3] for b in batch]

    return (node_batch,  pos_batch,
            E_total_batch, N_batch)

def plot_pred_vs_targ(y_test, y_pred):
    sort_idx = np.argsort(y_test.ravel())
    y_test_sorted = np.array(y_test).ravel()[sort_idx]
    y_pred_sorted = np.array(y_pred).ravel()[sort_idx]
    
    plt.figure(figsize=(10, 4))
    plt.scatter(y_test_sorted, y_pred_sorted, s=5, alpha=0.5)
    plt.plot([y_test_sorted.min(), y_test_sorted.max()],
             [y_test_sorted.min(), y_test_sorted.max()], 'r--')
    plt.xlabel('Target Energy')
    plt.ylabel('Predicted Energy')
    plt.title('KRR: Predicted vs Target Energy')
    plt.legend()
    plt.savefig('PredictedVSTarget.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_reg(y_test, y_pred):
    plt.figure(figsize=(10,4))
    plt.plot(y_test, color='green', label='Target Energy')
    plt.plot(y_pred, color='red', label='Predicted Energy')
    plt.ylabel('Energy / kJ/mol')
    plt.xlabel('Molecule')
    plt.title('KRR: Regression Plot')
    plt.legend()
    plt.savefig('RegressionPlot.png', dpi=150, bbox_inches='tight')
    plt.close()



def main():
    # plot energies per molecule
    mol_filename = 'water_set_1000.xyz'

    # 1. Load or generate molecules
    if os.path.exists(mol_filename):
        print(f"Found {mol_filename}; loading data (recomputing LJ energy).")
        molecules, position_list = load_molecules_from_xyz(mol_filename)
    else:
        print(f"Generating new training data...")
        molecules, position_list = generate_molecules(n_h2o=1000, filename=mol_filename)
        
    # 2. Convert to tensors
    print("Converting to tensors...")
    samples, node_t, targets_E, energy_per_molecule = molecules_to_tensors(molecules, device)
    
    #-------------CREATING FEATURE REPRESENTION-----------------
    # finish 16/06/2026, 

    molecule_repr = np.array([])
    pos_repr = interatomic_dist(position_list)

    # collecting node representations and converting to array
    node_repr = []
    for pos, els, E_total, F_atom in molecules: 
        row = np.array([log_primes[el] for el in els])
        node_repr.append(row)

    node_repr = np.array(node_repr)
    # Debugging, checking shapes of representations for concatenation
   
    print(f'Shape of node representation: {node_repr.shape}')
    print(f'Shape of position representation: {pos_repr.shape}')
    
    molecule_repr = np.hstack((node_repr, pos_repr))
  
    # debugging, seeing shape of represnetiation 
    print(f'Shape of representation is {(molecule_repr.shape)}')
    print(f'First node feature: {molecule_repr[0][0]}')

    

    # Add target energies to end of representations 
    print(f'Shape of target energies: {len(targets_E)}')
    
    dataset = np.hstack((molecule_repr, np.array(targets_E).reshape(1000,1)))
    target_energies = np.array(targets_E).reshape(1000,1)
    print(f'Shape of final dataset: {dataset.shape}')
    
    # in training set : 80 % feature representation and calculated lj energies
    
    # in testing set : 20 % feature representation and calculated lj energies
    X_train, X_test, y_train, y_test = train_test_split(molecule_repr, target_energies, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Generating kernel ridge regression model")
    # 4. Initialize model and optimizer
    model = KernelRidge(alpha=0.1, kernel='laplacian', gamma=0.1)
    
    # 5. Fit model to training set
    model.fit(X_train, y_train)

    # 6. Make predictions of energy using model 
    y_pred = model.predict(X_test)

    # 7. Evaluate model 
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error of KRR: {mse}')

    # 8. Single‑molecule test (H2O equilibrium)
    pos_test = np.array([[0.,0.,0.],
                         [0.96,0.,0.],
                         [-0.48,0.83,0.]])
    
    els = ['O','H','H']
    pos_test_repr = np.array([[np.linalg.norm(pos_test[0] - pos_test[1]),
                            np.linalg.norm(pos_test[0] - pos_test[2])]])
    node_test_repr = np.array([[log_primes[el] for el in els]])
    test_feat = np.hstack((node_test_repr, pos_test_repr))

    total_E_np = float(model.predict(test_feat).flatten()[0])
    target_E = lj_potential(pos_test, 0.1)
    print(f'total_e_np is: {total_E_np}')

    plot_pred_vs_targ(y_test, y_pred)
    plot_reg(y_test, y_pred)

    node_feats = np.array([log_primes[el] for el in els])[:, None]
    node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)


    pos_t = torch.tensor(pos_test, dtype=torch.float32, device=device, requires_grad=True)

   

    print("\n=== Single‑molecule test (H2O, equilibrium) ===")
    print(f"Target total energy (LJ):        {target_E:.6f}")
    print(f"Model total energy:              {total_E_np:.6f}")
   
    print(f"Model energy error (abs):        {abs(target_E - total_E_np):.6f}")
   
if __name__ == "__main__":
    main()
