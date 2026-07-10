import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.autograd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os

from ase import Atoms 
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.geometry import get_distances


"""
WedgeForceField:
- Variable‑size molecules (2, 3, ..., N atoms).
- Each molecule → total energy + per‑atom forces (N, 3).
- Uses CUDA if available.
- Train/test: 80/20 via DataLoader.
- Training: 50% energy vs LJ, 50% per‑atom LJ forces.
"""


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



def get_edge_features(atoms: Atoms, elements: list, cutoff: float = 8.0):

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


# MODEL: energy + per‑atom forces
class WedgeMLP(nn.Module):
    """
    forward_energy: scalar E_total.
    energy_and_forces: (E_total, forces) with forces = -∇E w.r.t. pos_t.
    """
    def __init__(self):
        super().__init__()

        self.node_net = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(), 
            nn.Linear(64,64),nn.LeakyReLU(), 
            nn.Linear(64,4), nn.LeakyReLU(),
             nn.Linear(4,1)
        )
        self.edge_net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), 
            nn.Linear(64,64), nn.LeakyReLU(), 
            nn.Linear(64,4),nn.LeakyReLU(),
            nn.Linear(4,1)
        )
        self.triplet_net = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(), 
            nn.Linear(64,64),nn.LeakyReLU(), 
            nn.Linear(64,4), nn.LeakyReLU(),
             nn.Linear(4,1)
        )



        self.to(device)

    def forward_energy(self, node_feats, edge_feats, triplets):
        node_E = self.node_net(node_feats).sum()
        edge_E = self.edge_net(edge_feats).sum()
        if len(triplets) > 0:
            edge_E = edge_E + self.triplet_net(triplets).sum()
        return node_E + edge_E

    def energy_per_atom(self, node_feats, edge_feats, triplets):
        total_E = self.forward_energy(node_feats, edge_feats, triplets)
        N = node_feats.size(0)
        return total_E / N

    def energy_and_forces(self, node_feats, edge_feats, triplets, pos_t):
        """
        pos_t: (N, 3), requires_grad=True.
        Returns:
            total_E: scalar
            forces: (N, 3) = -∂E/∂pos_t
        """
        with torch.enable_grad():
            total_E = self.forward_energy(node_feats, edge_feats, triplets)

            # Explicitly require_grad on pos_t if it ever got detached
            if pos_t.grad is not None:
                pos_t.grad.zero_()

            # Compute gradient
            grad_outputs = torch.ones_like(total_E)
            grad_list = torch.autograd.grad(
                outputs=total_E,
                inputs=pos_t,
                grad_outputs=grad_outputs,
                create_graph=False,
                allow_unused=True
            )

            if grad_list[0] is None:
                # Something is wrong; fall back to zeros
                forces = torch.zeros_like(pos_t)
            else:
                forces = -grad_list[0]   # F = -∇E

        return total_E, forces
    
    def save(self, filename='model_2.pt'):

        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")


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

        samples.append((node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N))
        targets_E.append(E_total)

    return samples, targets_E


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

    return (node_batch, edge_batch, triplet_batch, pos_batch,
            E_total_batch, F_batch, N_batch)


def train_model_energy(model, train_loader, test_loader, optimizer, epochs):
    train_losses = []
    test_losses = []

    last_epoch_loss = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for (node_batch, edge_batch, triplet_batch, pos_batch,
             E_total_tar, F_tar, N_batch) in train_loader:

            batch_size = len(E_total_tar)
            optimizer.zero_grad()

            loss_E_tot = 0.0

            for i in range(batch_size):
                node_i = node_batch[i]
                edge_i = edge_batch[i]
                trip_i = triplet_batch[i]
                pos_i = pos_batch[i].clone().detach().requires_grad_(True)
                N = N_batch[i]

                # 1. Total energy loss (LJ)
                total_E_pred = model.forward_energy(node_i, edge_i, trip_i)
                loss_E = (total_E_pred - E_total_tar[i])**2
                loss_E_tot += loss_E

            loss_E_avg = loss_E_tot / batch_size
        
           
            total_loss = loss_E_avg 
            #print("Running batch loss tracker. E_loss:", loss_E_avg) 
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss

        avg_epoch_loss = epoch_loss /len(train_loader)
        print("Running epoch loss tracker:", avg_epoch_loss, "last epoch's loss:", last_epoch_loss) 
        train_losses.append(avg_epoch_loss.detach().cpu().item())
        last_epoch_loss = avg_epoch_loss

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (node_batch, edge_batch, triplet_batch, pos_batch,
                 E_total_tar,F_tar, N_batch) in test_loader:
                batch_size = len(E_total_tar)
                batch_loss = 0.0
                for i in range(batch_size):
                    node_i = node_batch[i]
                    edge_i = edge_batch[i]
                    trip_i = triplet_batch[i]
                    pos_i = pos_batch[i]
                    N = N_batch[i]

                    total_E_pred = model.forward_energy(node_i, edge_i, trip_i)
                    loss_E = (total_E_pred - E_total_tar[i])**2

                    batch_loss += loss_E

                test_loss += batch_loss / batch_size

            test_loss /= len(test_loader)
            test_losses.append(test_loss.detach().cpu().item())
            print(f'Test loss: {test_loss}')
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss:.4f}, Test Loss = {test_loss:.4f}")

    model.save('model_dropout.pt')
    return train_losses, test_losses


# VISUALIZATION
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss for energy')
    plt.plot(test_losses,  label='Test Loss for energy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP on xTB Training (H2O for energy)')
    plt.legend()
    plt.savefig('mlp_xtb.png', dpi=150, bbox_inches='tight')
    plt.close()


# MAIN EXECUTION
def main():
    mol_filename = 'water_test.xyz'
    model_filename = 'mlp_xtb.pt'

    # 1. Load or generate molecules
    if os.path.exists(mol_filename):
        print(f"Found {mol_filename}; loading data (recomputing LJ energy).")
        molecules = load_from_extxyz(mol_filename)
    else:
        print(f"Generating new training data...")
        molecules = load_from_extxyz(filename='water_test.xyz')
    
    # 2. Convert to tensors
    print("Converting to tensors...")

    samples, targets_E = molecules_to_tensors(molecules, device)


    print(f'The structure of molecules is: {molecules}')
    print(f'targets_E: {targets_E}')

    # 3. Create Dataset + train/test split (80/20)
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

    print("Loading/generating model")
    # 4. Initialize model and optimizer
    model = WedgeMLP()
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, map_location=device))
        print(f"Training model loaded from {model_filename}")
    else:
        print("Training new model...")

    lr = 1e-6
    epochs = 5
    print("Learning rate:", lr)
    optimizer = optim.Adam(model.parameters(), lr)
    train_losses, test_losses = train_model_energy(model, train_loader, test_loader, optimizer, epochs)
    #plot_losses(train_losses, test_losses)

    print('Done training ')

    # leave for now 
    # 5. Single‑molecule test (H2O equilibrium)
    pos_test = read('water_test.xyz')
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


if __name__ == "__main__":
    main()