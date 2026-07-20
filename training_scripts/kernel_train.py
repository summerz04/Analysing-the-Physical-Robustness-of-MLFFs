import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.autograd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# kernel ridge model 
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


from ase import Atoms 
from ase.io import read
from ase.geometry import get_distances

"""Kernel ridge regression can only perform energy predictions, and cannot predict forces and cannot integrate for forces. """

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# PRIME ENCODING, labelling elements
prime_assign = {'H': 2, 'O': 7, 'F': 11, 'N': 13, 'C': 17}
log_primes = {el: np.log(p) for el, p in prime_assign.items()}

#  read traj files to use 
def get_molecular_distances(atoms: Atoms, molecule_indices:list, mic: bool = True):
    """"""

    # get centre of mass for each molecule 
    coms = []
    for mol in molecule_indices:
        com = atoms[mol].get_center_of_mass()
        coms.append(com)
    
    coms = np.array(coms)

    # calculate distance matrix 
    distances = np.zeros((len(coms), len(coms)))
    for i in range(len(coms)):
        for j in range(len(coms)):
            dist_vec, dist_len = get_distances(coms[i], coms[j], cell=atoms.cell, pbc=atoms.pbc)
            distances[i, j] = dist_len[0]

    return distances 



def get_edge_features(atoms: Atoms, elements: list, cutoff: float = 4.0):

    """Edge tokens: [log(p_i*p_j), Q_ij] for i < j.
    Edge features are stacked with log prods and intermolecular distances"""

    # get atoms object for each frame 

    N = len(elements)
    if N < 2:
        return np.empty((0, 2))

    i_idx, j_idx = np.triu_indices(N, k=1)

    # minimum image convention for each pair 
    mic_dists = np.array([
        atoms.get_distances(int(i), int(j), mic=True)[0] for i, j in zip(i_idx, j_idx)
    ])
    
    # applying cut off distance 
    i_idx = i_idx[mic_dists < cutoff]
    j_idx = j_idx[mic_dists < cutoff]

    mic_dists = mic_dists[mic_dists < cutoff]

    log_prods = np.array([log_primes[elements[i]] + log_primes[elements[j]]
                         for i, j in zip(i_idx, j_idx)])
    stacked_dists = np.stack([log_prods, mic_dists], axis=-1)

    print(f'Shape of edge features: {stacked_dists.shape}')
    return stacked_dists


def wedge_product(edge_feats):
    """Wedge product: wedge(e1,e2) = t1*q2 - q1*t2."""
    E = len(edge_feats)
    if E < 2:
        return np.array([])

    if edge_feats.ndim == 1:
        edge_feats = edge_feats.reshape(1, 2)

    t, q = edge_feats[:, 0], edge_feats[:, 1]
    return t[:-1] * q[1:] - q[:-1] * t[1:]


#--------------------------------
# kernel ridge regression model
#--------------------------------




# no need to generate dataset, reading from extxyz file
def load_from_extxyz(filename):
    """
    Reads extended xyz files from ASE simulations
    and returns molecule data for MLFF training
    
    Parameters
    ----------
    filename: string 
        The name of the extended .xyz file
        
    Returns
    --------
    molecules: tuple 
        (pos, elements, E_tot, F_atom)
        A tuple of atomic coordinates ((n, 3) np array), elements (str list),
        total potential energy and atomic forces ((n, 3) np array)"""

    frames = read(filename, index=':')

    molecules = []
    for atoms in frames:
        # extract elements 
        elements = atoms.get_chemical_symbols()

        # extract positions, variables to match models 
        pos = atoms.get_positions()

        # extract total energy 
        target_Etot = atoms.get_potential_energy()

        # extract atomic forces 
        F_atom = atoms.get_forces()

        print(elements[0])
        molecules.append((atoms, pos, elements, target_Etot, F_atom)) # same format as model, with added atoms object per frame

    print(f'Loaded molecules from {filename}')
    return molecules


def molecules_to_tensors(molecules, device):
    """
    molecules: list of (pos, els, E_total, F_atom).

    Returns list of (node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N_atoms)
    for each molecule.
    """
    samples = []
    targets_E = []


    for atoms, pos, els, E_total, F_atom in molecules:
        N = len(els)

        # 1. Node features
        node_feats = np.array([log_primes[el] for el in els])[:, None]
        node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)


        # 2. Edge features
        
        if N >= 2:
            edge_feats = get_edge_features(atoms, els)
            print(f'shape of edge_feats = {edge_feats.shape}')
            if len(edge_feats) == 2:
                edge_feats = edge_feats.reshape(1, 2)
            if edge_feats.ndim == 1:
                edge_feats = np.zeros((0, 2))
        else:
            edge_feats = np.zeros((0, 2))
        edge_t = torch.tensor(edge_feats, dtype=torch.float32, device=device)
        print(f'shape of edge features per frame: {edge_feats.shape}')

        # 3. Triplets
        wedges = wedge_product(edge_feats)
        if len(wedges) > 0:
            trip_t = torch.tensor(wedges[:, None], dtype=torch.float32, device=device)
        else:
            trip_t = torch.zeros(1, 1, dtype=torch.float32, device=device)

        # 4. Positions (requires_grad=True)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device, requires_grad=True)

        
        # 5. LJ forces
        F_atom_t = torch.tensor(F_atom, dtype=torch.float32, device=device)


        #6. atoms object cell length
        cell_t = torch.tensor(atoms.cell.lengths(), dtype=torch.float32, device=device)
        
        samples.append((node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N, els, cell_t))
        targets_E.append(E_total)

    return samples, targets_E


# everything here should be in pytorch 
def edge_features_torch(pos_t, elements, cell_size, cutoff: float = 4.0):
    """For autograd to generate forces"""
    N = len(elements)
    pairs = torch.triu_indices(row=N, col=N, offset=1) # unique pairs of atom indices

    
    i_idx = pairs[0]
    j_idx = pairs[1]

    # minimum image convention distance 
    raw_diff = pos_t[i_idx] - pos_t[j_idx]
    
    diff = raw_diff - cell_size * torch.round(raw_diff / cell_size)

    dists = torch.norm(diff, dim=-1)
    mask = dists < cutoff 
    dists = dists[mask] # applying cutoff for meaningful distances
    
    log_prods = torch.tensor([log_primes[elements[i]] + log_primes[elements[j]]
                         for i, j in zip(i_idx.tolist(), j_idx.tolist())], dtype=pos_t.dtype)
    
    log_prods = log_prods[mask]
    print(f'shape of log prds: {log_prods.shape}')
    edge_features = torch.stack((log_prods, dists), dim=1)

    print(f'Shape of edge features: {edge_features.shape}')
    return edge_features
    
def triplet_features_torch(edge_feats):
    """Differentiable version: wedge(e1,e2) = t1*q2 - q1*t2."""
    if edge_feats.shape[0] < 2:
        return torch.zeros(1, 1, dtype=edge_feats.dtype, device=edge_feats.device)
    t, q = edge_feats[:, 0], edge_feats[:, 1]
    wedges = t[:-1] * q[1:] - q[:-1] * t[1:]
    return wedges.unsqueeze(-1)

# DATASET CLASS + COLLATE
class EnergyDataset(Dataset):
    """
    Each item = (node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N_atoms).
    """
    def __init__(self, samples):
        self.samples = samples

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
    edge_batch = [b[1] for b in batch]
    triplet_batch = [b[2] for b in batch]
    pos_batch = [b[3] for b in batch]
    E_total_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
    F_batch = [b[5] for b in batch]
    N_batch = [b[6] for b in batch]
    elements_batch = [b[7] for b in batch]
    cellsize_batch =[b[8] for b in batch]

    return (node_batch, edge_batch, triplet_batch, pos_batch,
            E_total_batch, F_batch, N_batch, elements_batch, cellsize_batch)


def train_model_energy_forces(model, train_loader, test_loader, optimizer, epochs):
    # monitor energy and forces separately
    train_energy_losses = []
    train_force_losses = []

    test_energy_losses = []
    test_force_losses = []

    last_epoch_loss = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_energy_loss = 0.0
        epoch_force_loss = 0.0

        for (node_batch, edge_batch, triplet_batch, pos_batch,
             E_total_tar, F_tar, N_batch, elements_batch, cellsize_batch) in train_loader:

            batch_size = len(E_total_tar)
            optimizer.zero_grad()

            batch_energy_loss = 0.0
            batch_force_loss = 0.0

            for i in range(batch_size):
                node_i = node_batch[i]
                cell_i = cellsize_batch[i]
                F_tar_i = F_tar[i]
                elements_i = elements_batch[i]
                
                pos_i = pos_batch[i].clone().detach().requires_grad_(True)
                
                N = N_batch[i]
                # computing edge features in training to access forces
                edge_i = edge_features_torch(pos_i, elements_i, cell_i)

                # computing triplet features to access forces
                trip_i = triplet_features_torch(edge_i)

                # 1. energy loss 
                E_pred = model.forward_energy(node_i, edge_i, trip_i)
                loss_E = (E_pred - E_total_tar[i])**2
                

                # 2. Force loss 
                grads = torch.autograd.grad(E_pred, pos_i, create_graph=True)[0]
                F_pred = -grads
                loss_F = ((F_pred - F_tar_i)**2).mean()
            
                batch_energy_loss += loss_E
                batch_force_loss += loss_F
                

            energy_loss = batch_energy_loss / batch_size
            force_loss = batch_force_loss / batch_size 

            total_loss = 0.5*energy_loss + 0.5*force_loss
            #print("Running batch loss tracker. E_loss:", loss_E_avg) 
            total_loss.backward()
            optimizer.step()

            epoch_energy_loss += energy_loss.item()
            epoch_force_loss += force_loss.item()

        avg_energy_loss = epoch_energy_loss / len(train_loader)
        avg_force_loss = epoch_force_loss / len(train_loader)

        train_energy_losses.append(avg_energy_loss)
        train_force_losses.append(avg_force_loss)

        # Evaluation
        model.eval()

        test_epoch_energy = torch.tensor(0.0, device=device) 
        test_epoch_force = torch.tensor(0.0, device=device)
        
        with torch.enable_grad():
            # force evaluation 
            for (node_batch, edge_batch, triplet_batch, pos_batch,
                    E_total_tar,F_tar, N_batch, elements_batch, cellsize_batch) in test_loader:
                batch_size = len(E_total_tar)

                batch_energy_loss = 0.0
                batch_force_loss = 0.0
                for i in range(batch_size):
                    node_i = node_batch[i]
                    cell_i = cellsize_batch[i]
                    #edge_i = edge_batch[i]
                    #trip_i = triplet_batch[i]
                    F_tar_i = F_tar[i]
                    
                    elements_i = elements_batch[i]
                    N = N_batch[i]

                    pos_i = pos_batch[i].clone().detach().requires_grad_(True)
                    edge_i = edge_features_torch(pos_i, elements_i, cell_i)
                    trip_i = triplet_features_torch(edge_i)
                    E_pred = model.forward_energy(node_i, edge_i, trip_i)
                    loss_E = (E_pred - E_total_tar[i])**2

                    grads = torch.autograd.grad(E_pred, pos_i, create_graph=False)[0]
                    F_pred = -grads
                    loss_F = ((F_pred - F_tar_i)**2).mean()
                    batch_energy_loss += loss_E
                    batch_force_loss += loss_F
                    
                test_epoch_energy += batch_energy_loss / batch_size
                test_epoch_force += batch_force_loss / batch_size

            avg_test_energy = test_epoch_energy / len(test_loader)
            avg_test_force = test_epoch_force / len(test_loader)

            test_energy_losses.append(avg_test_energy.detach().cpu().item())
            test_force_losses.append(avg_test_force.detach().cpu().item())

            print(f'testing energy loss :{test_energy_losses}')
            print(f'testing force loss :{test_force_losses}')

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_energy_loss:.4f}")
            print(f"Epoch {epoch}: Train Loss = {avg_force_loss:.4f}")

    model.save('mlp.xtb.pt')
    return train_energy_losses, test_energy_losses, train_force_losses, test_force_losses


# VISUALIZATION
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

def plot_energy_losses(train_energy_losses, test_energy_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_energy_losses, label='Training loss')
    plt.plot(test_energy_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on xTB Training (H2O for energy)')
    plt.legend()
    plt.savefig(f'./training_figs/{model_name}_xtb_energy.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_force_losses(train_force_losses, test_force_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_force_losses, label='Training loss')
    plt.plot(test_force_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on xTB Training (H2O for forces)')
    plt.legend()
    plt.savefig(f'./training_figs/{model_name}_xtb_forces.png', dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------
# Setting up training for all models 
#-------------------------------------


def train_all(model_name, model_class, train_loader, test_loader, epochs=100, lr=1e-4):
    print(f'Training {model_name} model...')
    model = model_class()
    model_filename = f'{model_name}_xtb.pt'

    # 1. Load or generate molecules
    if os.path.exists(model_filename):
        print(f"Found {model_filename}; loading model.")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr)

    # add message passing 

    train_energy_losses, test_energy_losses, train_force_losses, test_force_losses = train_model_energy_forces(model, train_loader, test_loader, optimizer, epochs)
    model.save(model_filename)

    return {
        'model': model,
        'train_energy': train_energy_losses,
        'test_energy': test_energy_losses,
        'train_force': train_force_losses,
        'test_force': test_force_losses,
    }

def plot_comparison(results, key, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10,5))
    for name, result in results.items():
        ax.plot(result[key], label=name)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} comparison across models')
    plt.legend()
    plt.savefig(f'./training_figs/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print('Finished plotting')

# MAIN EXECUTION
def main():
    mol_filename = '../water_test.xyz'

    # 1. Load or generate molecules
    if os.path.exists(mol_filename):
        print(f"Found {mol_filename}; loading data")
        molecules = load_from_extxyz(mol_filename)
    else:
        print(f"Can't find extended .xyz file")
        
    
    # 2. Convert to tensors
    print("Converting to tensors...")

    samples, targets_E = molecules_to_tensors(molecules, device)

    X = []
    # create master numpy array containing all features 
    for node_t, edge_t, trip_t, pos_t, E, F, N, els, cell in samples:
        features = np.concatenate([node_t.mean(0).cpu().numpy(),
                                   edge_t.mean(0).cpu().numpy(),
                                   trip_t.mean(0).cpu().numpy(),
                                   np.array([N])])
        X.append(features)

    X = np.array(X)
    y = np.array(targets_E)

    print(f'Shape of feature representation: {X.shape}')
    print(f'Shape of target energies: {y.shape}')

    # splitting data into training and testing sets
    print('Splitting into training and testing 80/20')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardise features 
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Create Dataset + train/test split (80/20)
    print('Creating dataset...')
    dataset = EnergyDataset(samples)
    N = len(dataset)
    indices = torch.randperm(N)
    train_size = int(0.8 * N)
    train_idx = indices[:train_size]
    test_idx  = indices[train_size:]

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        sampler=SubsetRandomSampler(train_idx)
    )

    test_loader  = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        sampler=SubsetRandomSampler(test_idx)
    )

    print("Beginning to train KRR...")

    # create regression model 
    model = KernelRidge(alpha=0.1, kernel='laplacian', gamma=0.1)

    # fit model to training set
    model.fit(X_train, y_train)

    # make predictions of energy and forces using model 
    y_pred = model.predict(X_test)

    # evaluate model 
    mse = mean_squared_error(y_test, y_pred)

   # plot regression plots 
    plot_pred_vs_targ(y_test, y_pred)
    plot_reg(y_test, y_pred)

    print(f'MSE: {mse}')

    

    # leave for now 
    # 5. Single‑molecule test (H2O equilibrium)
    """pos_test = read('water_test.xyz')
    els = ['O','H','H']
    
    # extract real energy from xyz 
    node_feats = np.array([log_primes[el] for el in els])[:, None]
    node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)

    edge_feats = get_edge_features(els, pos_test)[0]
    

    edge_t = torch.tensor(edge_feats, dtype=torch.float32, device=device)

    wedges = wedge_product(edge_feats)
    if len(wedges) > 0:
        trip_t = torch.tensor(wedges[:, None], dtype=torch.float32, device=device)
    else:
        trip_t = torch.zeros(1, 1, dtype=torch.float32, device=device)

    pos_t = torch.tensor(pos_test, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    with torch.no_grad():
        total_E_pred = model.forward_energy(node_t, edge_t, trip_t)
        E_per_atom_pred = model.energy_per_atom(node_t, edge_t, trip_t)
        total_E_actual, forces_pred = model.energy_and_forces(node_t, edge_t, trip_t, pos_t)

    total_E_np = total_E_pred.cpu().item()
    E_per_atom_np = E_per_atom_pred.cpu().item()
    forces_np = forces_pred.cpu().numpy()

    print("\n=== Single‑molecule test (H2O, equilibrium) ===")
    print(f"Target total energy (LJ):        {target_E:.6f}")
    print(f"Model total energy:              {total_E_np:.6f}")
    print(f"Model E_per_atom:                {E_per_atom_np:.6f}")
    print(f"Model energy error (abs):        {abs(target_E - total_E_np):.6f}")

    print("\nPredicted per‑atom forces (model):")
    for i, f_pred in enumerate(forces_np):
        print(f"  Atom {i}: {f_pred[0]:8.5f}, {f_pred[1]:8.5f}, {f_pred[2]:8.5f}")

    print(f"\nModel force balance (sum): {forces_np.sum(axis=0)}")
    print(f"LJ force balance (sum):    {target_F.sum(axis=0)}")

    F_rmse = np.sqrt(((forces_np - target_F)**2).mean())
    print(f"\nForce RMSE (per‑atom): {F_rmse:.6f}")
"""

if __name__ == "__main__":
    main()