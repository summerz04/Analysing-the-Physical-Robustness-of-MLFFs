
import torch 
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes
import numpy as np

# importing models 
from master_training import WedgeMLP

# PRIME ENCODING, labelling elements
prime_assign = {'H': 2, 'O': 7, 'F': 11, 'N': 13, 'C': 17}
log_primes = {el: np.log(p) for el, p in prime_assign.items()}

# edge features


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

# triplet features
def triplet_features_torch(edge_feats):
    """Differentiable version: wedge(e1,e2) = t1*q2 - q1*t2."""
    if edge_feats.shape[0] < 2:
        return torch.zeros(1, 1, dtype=edge_feats.dtype, device=edge_feats.device)
    t, q = edge_feats[:, 0], edge_feats[:, 1]
    wedges = t[:-1] * q[1:] - q[:-1] * t[1:]
    return wedges.unsqueeze(-1)

class ModelCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, device='cpu', cutoff=4.0):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cutoff = cutoff
    
    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        
        super().calculate(atoms, properties, system_changes)
        
        # get positions of atoms object
        pos = torch.tensor(atoms.get_positions(), 
                            dtype=torch.float32, 
                            device=self.device, 
                            requires_grad=True)
        
        # get cell 
        cell = torch.tensor(
            atoms.cell.lengths(),
            dtype=torch.float32,
            device=self.device)
        
        # get elements
        elements = atoms.get_chemical_symbols()

        # encode to get node features
        node_feats = np.array(
            [[log_primes[e]] for e in elements],
            dtype=np.float32)
        node_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device)

        
        # get edge features
        edge_t = edge_features_torch(pos, elements, cell, cutoff=self.cutoff)

        # get triplet features
        triplet_t = triplet_features_torch(edge_t)

        # calculate energy 
        energy = self.model.forward_energy(node_t, edge_t, triplet_t)

        forces = -torch.autograd.grad(energy, pos, create_graph=False, retain_graph=False)[0]

        # saves results in dictionary
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.detach().cpu().numpy()

        
#-----------------------
# testing calculator works
#-----------------------

# create model 
model = WedgeMLP()

# load model 
model.load_state_dict(torch.load('MLP_xtb.pt'))

# create calculator 
calc = ModelCalculator(model, device='cpu', cutoff=4.0)

# load xyz file to give calculator the structure 
atoms = read('water_test.xyz')

atoms.calc = calc

# get energy 
print(f'energy: {atoms.get_potential_energy()}')

# get forces
print(f'forces: {atoms.get_forces()}')

# checks
print("\nChecks:")
print("Number of atoms:", len(atoms))
