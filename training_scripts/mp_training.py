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


from torch_geometric.nn import MessagePassing
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


#---------------------------
# messaging passing
#---------------------------
class MoleculeMessagePassing(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super(MoleculeMessagePassing, self).__init__(aggr='add')
        
        # 1. encode edge features
        self.encode_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. combining node and edge features to create messages
        self.message_pass = nn.Sequential(
        nn.Linear(node_dim + hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. update node representations 
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    # forward for MESSAGE PASSING
    def forward(self, x, edge_index, edge_attr):
        embedded_edge = self.encode_edge(edge_attr)
        return self.propagate(edge_index, x=x, edge_emb=embedded_edge)
    
    # combines neighbour node features with edge features (DISTANCE)
    def message(self, x_j, edge_emb):
        # x_j are the node features of neighbours 
        return self.message_pass(torch.cat([x_j, edge_emb], dim=-1))
    
    # return node representations after aggregation
    def update(self, aggregated_out, x):
        return x + self.update_mlp(torch.cat([x, aggregated_out], dim=-1))

# Message passing algorithm + mlp: energy + per atom forces
class MP_MLP(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.GNN = MoleculeMessagePassing(node_dim=1,
                                        edge_dim=2, 
                                        hidden_dim=32,
                                        out_dim=1)

        self.node_net = nn.Sequential(
            nn.Linear(3, 64), nn.Mish(), 
            nn.Linear(64, 64), nn.Mish(),
            nn.Linear(64, 4), nn.Mish(), 
            nn.Linear(4, 1)
        )
       
        self.to(device)

    def forward_energy(self, node_feats, edge_feats, edge_idx, trip_feats):

        # message passing
        new_node_feats = self.GNN(node_feats, edge_idx, edge_feats)
        
        # combine all features into combined feats
        combined_feats = torch.cat([node_feats, new_node_feats], dim=1)
        
        # utilise triplet features, find average and tie to each instance
        # information loss using triplets because of the structure of message passing, 
        # hard to index triplet values singularly
        trip = trip_feats.mean(dim=0, keepdim=True)
        trip_all = trip.expand(combined_feats.shape[0], -1)

        final_input = torch.cat([combined_feats, trip_all], dim=1)
        node_E = self.node_net(final_input).sum()
        return node_E
    
    def save(self, filename='message_pass.pt'):

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

    # apply mask to indexes and distances
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    dists = dists[mask] 
    
    log_prods = torch.tensor([log_primes[elements[i]] + log_primes[elements[j]]
                         for i, j in zip(i_idx.tolist(), j_idx.tolist())], dtype=pos_t.dtype)
    
    edge_features = torch.stack((log_prods, dists), dim=1)
    edge_idx = torch.stack([i_idx, j_idx], dim=0)
    
    return edge_features, edge_idx
    
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


def train_model_energy_forces(model, train_loader, test_loader, epochs, optimizer):
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
                edge_i, edge_index_i = edge_features_torch(pos_i, elements_i, cell_i)

                # computing triplet features to access forces
                trip_i = triplet_features_torch(edge_i)

                # 1. energy loss 
                E_pred = model.forward_energy(node_i, edge_i, edge_index_i, trip_i)
                loss_E = (E_pred - E_total_tar[i])**2
                

                # 2. Force loss 
                grads = torch.autograd.grad(E_pred, pos_i, create_graph=True)[0]
                F_pred = -grads
                loss_F = ((F_pred - F_tar_i)**2).mean()
            
                batch_energy_loss += loss_E
                batch_force_loss += loss_F
                

            energy_loss = batch_energy_loss / batch_size
            force_loss = batch_force_loss / batch_size 

            total_loss = energy_loss + (1000*force_loss)
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
                    edge_i, edge_index_i = edge_features_torch(pos_i, elements_i, cell_i)
                    trip_i = triplet_features_torch(edge_i)
                    E_pred = model.forward_energy(node_i, edge_i, edge_index_i, trip_i)
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
def plot_energy_losses(train_energy_losses, test_energy_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_energy_losses, label='Training loss')
    plt.plot(test_energy_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on xTB Training (H2O for energy)')
    plt.legend()
    plt.savefig(f'../training_figs/{model_name}_xtb_energy.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_force_losses(train_force_losses, test_force_losses, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_force_losses, label='Training loss')
    plt.plot(test_force_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on xTB Training (H2O for forces)')
    plt.legend()
    plt.savefig(f'../training_figs/{model_name}_xtb_forces.png', dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------
# Setting up training for all models 
#-------------------------------------


def train_all(model_name, model_class, train_loader, test_loader, epochs=250, lr=1e-5):
    print(f'Training {model_name} model...')
    model = model_class()
    model_filename = f'{model_name}_xtb.pt'

    # 1. Load or generate molecules
    if os.path.exists(model_filename):
        print(f"Found {model_filename}; loading model.")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr)
    # add message passing 

    train_energy_losses, test_energy_losses, train_force_losses, test_force_losses = train_model_energy_forces(model, train_loader, test_loader, epochs, optimizer)
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
    plt.savefig(f'../training_figs/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print('Finished plotting')

# MAIN EXECUTION
def main():
    mol_filename = '../train_generation/shuffled_water_dataset.extxyz'

    # 1. Load or generate molecules
    if os.path.exists(mol_filename):
        print(f"Found {mol_filename}; loading data (recomputing LJ energy).")
        molecules = load_from_extxyz(mol_filename)
    else:
        print(f"Can't find extended .xyz file")
        
    
    # 2. Convert to tensors
    print("Converting to tensors...")

    samples, targets_E = molecules_to_tensors(molecules, device)

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

    print("Beginning to train MLP...")

    models = {
    'MP': MP_MLP
    }

    results ={}

    for name, mlff in models.items():
        results[name] = train_all(name, mlff, train_loader, test_loader)

   
        print(f'Done training {name}')

    for name, result in results.items():
        plot_energy_losses(result['train_energy'], result['test_energy'], name)
        plot_force_losses(result['train_force'], result['test_force'], name)
    
    # compare between plots
    #plot_comparison(results, key='test_energy', ylabel='Energy Loss', filename='energy_comparison.png')
    #plot_comparison(results, key='test_force', ylabel='Force Loss', filename='force_comparison.png')


if __name__ == "__main__":
    main()