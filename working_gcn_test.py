import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torch.autograd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os

from typing import Optional
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as utils 
from torch_geometric.utils import add_self_loops, degree, to_edge_index
from torch_geometric.data import Data


# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# PRIME ENCODING, labelling elements
prime_assign = {'H': 2, 'O': 7, 'F': 11, 'N': 13, 'C': 17}
log_primes = {el: np.log(p) for el, p in prime_assign.items()}

def generate_molecules(n_h2o=4000, n_hf=1500, filename='training_set.xyz'):
    """
    Returns [(pos, elements, E_total, F_atom)].
    Also writes filename as XYZ file.
    """
    molecules = []

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

    # HF
    for _ in range(n_hf):
        r_hf = np.random.normal(0.92, 0.1)
        pos = np.array([[0,0,0],
                        [r_hf, 0.01*np.random.randn(), 0.01*np.random.randn()]])
        E_total = lj_potential(pos, 0.1)
        F_atom = lj_forces(pos, 0.1)
        molecules.append((pos, ['H','F'], E_total, F_atom))

    # Write XYZ
    with open(filename, 'w') as f:
        for i, (pos, els, E, F) in enumerate(molecules):
            f.write(f"{len(els)}\n")
            f.write(f"mol_{i} E={E:.6f}\n")
            for j, el in enumerate(els):
                f.write(f"{el:2s} {pos[j][0]:12.6f} {pos[j][1]:12.6f} {pos[j][2]:12.6f}\n")
    print(f"Generated {len(molecules)} structures ÔåÆ {filename}")
    return molecules


def load_molecules_from_xyz(filename='training_set.xyz'):
    """
    Parse training_set.xyz ÔåÆ (pos, elements, E_total, F_atom).
    Recompute LJ energy; ignore E= from file.
    """
    molecules = []
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

    print(f"Loaded {len(molecules)} structures from {filename}")
    return molecules


# Scaled Lennard-Jones Potential
def lj_potential(pos, epsilon=0.1, scale=0.1):
    r2 = quadrance(pos) + 1e-10
    r6 = r2**3
    V = 4*epsilon * (r6**(-2) - r6**(-1))
    return np.triu(V, k=1).sum() * scale   # e.g. scale=0.01

def lj_forces(pos, epsilon=0.1, scale=0.1):
    """Numerical gradient of lj_potential ÔåÆ perÔÇæatom forces (N, 3)."""
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


class SimpleGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1,1) 

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        return F.relu(x)



    

# Linear MLP MODEL: energy + per atom forces
class MLP(nn.Module):
   
    def __init__(self):
        super().__init__()

        self.GCN = SimpleGCN()

        self.node_net = nn.Sequential(
            nn.Linear(2, 16), nn.LeakyReLU(), 
            nn.Linear(16, 8), nn.LeakyReLU(), 
            nn.Linear(8,4), nn.LeakyReLU(), nn.Linear(4,1)
        )
       
        self.to(device)

    def forward(self, node_feats, edge_feats):
        distance = quadrance(edge_feats.cpu())
        adjacency = torch.from_numpy(np.where(distance < 2, 1.0, 0.0)).float()
        adjacency = adjacency.to_sparse()
        edge_index, edge_attr = utils.to_edge_index(adjacency)
        edge_index = edge_index.to(device)
        new_node_feats = self.GCN(node_feats, edge_index)
        combined_feats = torch.cat([node_feats, new_node_feats], dim=1)
        node_E = self.node_net(combined_feats).sum()

        return node_E
    
    def save(self, filename='gcn_test.pt'):

        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}") 

def train_model(model, train_loader, test_loader, optimizer, epochs):
    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(epochs):
        # Training
        
        epoch_loss = 0.0

        for (node_batch, pos_batch,
             E_total_tar, N_batch) in train_loader:

           
            

            batch_size = len(E_total_tar)
            optimizer.zero_grad()

            loss_E_tot = 0.0
        

            for i in range(batch_size):
                node_i = node_batch[i]
                
              
                pos_i = pos_batch[i].clone().detach().requires_grad_(False)
                N = N_batch[i]

                # 1. Total energy loss (LJ)
                total_E_pred = model.forward(node_i, pos_i)
                loss_E = (total_E_pred - E_total_tar[i])**2
                loss_E_tot += loss_E

                

            loss_E_avg = loss_E_tot / batch_size
           

            # 50ÔÇô50 energyÔÇôforces
            total_loss = loss_E_avg 
            #print("Running batch loss tracker. E_loss:", loss_E_avg," F_loss:", loss_F_avg, " total loss:", total_loss) 
            #loss_tensor = torch.tensor(total_loss, device=device, requires_grad=True)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        print("Running epoch loss tracker:", avg_epoch_loss) 
        train_losses.append(avg_epoch_loss.detach().cpu().numpy())
    

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (node_batch,  pos_batch,
                 E_total_tar, N_batch) in test_loader:
                batch_size = len(E_total_tar)
                batch_loss = 0.0
                for i in range(batch_size):
                    node_i = node_batch[i]
                  
                    pos_i = pos_batch[i]
                    N = N_batch[i]

                    total_E_pred = model.forward(node_i, pos_i)
                    loss_E = (total_E_pred - E_total_tar[i])**2


                    batch_loss += loss_E 

                test_loss += batch_loss / batch_size

            test_loss /= len(test_loader)
            test_losses.append(test_loss.detach().cpu().numpy())
            print(f'Test loss: {test_loss}')
            

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss:.4f}, Test Loss = {test_loss:.4f}")

    model.save('gcn_test.pt')
    return train_losses, test_losses

def molecules_to_tensors(molecules, device):
    """
    molecules: list of (pos, els, E_total, F_atom).

    Returns list of (node_t, edge_t, trip_t, pos_t, E_total, F_atom_t, N_atoms)
    for each molecule.
    """
    samples = []
    targets_E = []

    for pos, els, E_total, F_atom in molecules:
        N = len(els)

        # 1. Node features
        node_feats = np.array([log_primes[el] for el in els])[:, None]
        node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)

       

        # 4. Positions (requires_grad=True)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device, requires_grad=True)


        samples.append((node_t, pos_t, E_total, N))
        targets_E.append(E_total)

    return samples, targets_E

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

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss for energy')
    plt.plot(test_losses,  label='Test Loss for energy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP test Training (LJÔÇæH2O/HF for energy)')
    plt.legend()
    plt.savefig('gcn_test.png', dpi=150, bbox_inches='tight')
    plt.close()

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

def main():
    mol_filename = 'training_set.xyz'
    model_filename = 'gcn_test.pt'

    # 1. Load or generate molecules
    if os.path.exists(mol_filename):
        print(f"Found {mol_filename}; loading data (recomputing LJ energy).")
        molecules = load_molecules_from_xyz(mol_filename)
    else:
        print(f"Generating new training data...")
        molecules = generate_molecules(n_h2o=4000, n_hf=1500, filename=mol_filename)
    

    # 2. Convert to tensors
    print("Converting to tensors...")
    samples, targets_E = molecules_to_tensors(molecules, device)

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
        shuffle=True,
        num_workers=0
    )
    test_loader  = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0
    )

    print("Loading/generating model")
    # 4. Initialize model and optimizer
    model = MLP()
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, map_location=device))
        print(f"Training model loaded from {model_filename}")
    else:
        print("Training new model...")

    lr = 1e-4
    epochs = 40
    print("Learning rate:", lr)
    optimizer = optim.Adam(model.parameters(), lr)
    train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, epochs)
    plot_losses(train_losses, test_losses)

    # 5. SingleÔÇæmolecule test (H2O equilibrium)
    pos_test = np.array([[0.,0.,0.],
                         [0.96,0.,0.],
                         [-0.48,0.83,0.]])
    els = ['O','H','H']
    target_E = lj_potential(pos_test, 0.1)
  

    node_feats = np.array([log_primes[el] for el in els])[:, None]
    node_t = torch.tensor(node_feats, dtype=torch.float32, device=device)


    pos_t = torch.tensor(pos_test, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    with torch.no_grad():
        total_E_pred = model.forward(node_t, pos_t)
       

    total_E_np = total_E_pred.cpu().item()
   

    print("\n=== SingleÔÇæmolecule test (H2O, equilibrium) ===")
    print(f"Target total energy (LJ):        {target_E:.6f}")
    print(f"Model total energy:              {total_E_np:.6f}")
   
    print(f"Model energy error (abs):        {abs(target_E - total_E_np):.6f}")



   


if __name__ == "__main__":
    main()

