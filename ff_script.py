import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.autograd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os


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


def quadrance(pos):
    """Squared distances Q_ij = ||r_i - r_j||^2."""
    return squareform(pdist(pos, 'sqeuclidean'))


def get_edge_features(elements, pos):
    """Edge tokens: [log(p_i*p_j), Q_ij] for i < j."""
    N = len(elements)
    if N < 2:
        return np.empty((0, 2))

    i_idx, j_idx = np.triu_indices(N, k=1)
    log_prods = np.array([log_primes[elements[i]] + log_primes[elements[j]]
                         for i, j in zip(i_idx, j_idx)])
    Q = quadrance(pos)[i_idx, j_idx]
    return np.stack([log_prods, Q], axis=-1)


def wedge_product(edge_feats):
    """Wedge product: wedge(e1,e2) = t1*q2 - q1*t2."""
    E = len(edge_feats)
    if E < 2:
        return np.array([])

    if edge_feats.ndim == 1:
        edge_feats = edge_feats.reshape(1, 2)

    t, q = edge_feats[:, 0], edge_feats[:, 1]
    return t[:-1] * q[1:] - q[:-1] * t[1:]


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


# MODEL: energy + per‑atom forces
class WedgeMLP(nn.Module):
    """
    forward_energy: scalar E_total.
    energy_and_forces: (E_total, forces) with forces = -∇E w.r.t. pos_t.
    """
    def __init__(self):
        super().__init__()

        self.node_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.edge_net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.triplet_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )

        self.to(device)

    def forward_energy(self, node_feats, edge_feats, triplets):
        node_E = self.node_net(node_feats).sum()
        edge_E = self.edge_net(edge_feats).sum()
        trip_E = 0.0
        if len(triplets) > 0:
            trip_E = self.triplet_net(triplets).sum()
        return node_E + edge_E + trip_E

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
    
    def save(self, filename='wedge_model.pt'):

        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")


# DATASET GENERATION
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
    print(f"Generated {len(molecules)} structures → {filename}")
    return molecules


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

        # 2. Edge features
        if N >= 2:
            edge_feats = get_edge_features(els, pos)[0]
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

def load_molecules_from_xyz(filename='training_set.xyz'):
    """
    Parse training_set.xyz → (pos, elements, E_total, F_atom).
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

# 50/50 TRAINING LOOP
def train_model_50_50(model, train_loader, test_loader, optimizer):
    train_losses = []
    test_losses = []

    last_epoch_loss = 0.0
    for epoch in range(100):
        # Training
        model.train()
        epoch_loss = 0.0

        for (node_batch, edge_batch, triplet_batch, pos_batch,
             E_total_tar, F_tar, N_batch) in train_loader:

            batch_size = len(E_total_tar)
            optimizer.zero_grad()

            loss_E_tot = 0.0
            loss_F_tot = 0.0

            for i in range(batch_size):
                node_i = node_batch[i]
                edge_i = edge_batch[i]
                trip_i = triplet_batch[i]
                pos_i = pos_batch[i].clone().detach().requires_grad_(True)
                N = N_batch[i]

                # 1. Total energy loss (LJ)
                total_E_pred = model.forward_energy(node_i, edge_i, trip_i)
                loss_E = (total_E_pred - E_total_tar[i])**2
                loss_E_tot += loss_E.item()

                # 2. Per‑atom force loss (LJ forces)
                total_E_actual, F_pred = model.energy_and_forces(node_i, edge_i, trip_i, pos_i)
                F_tar_i = F_tar[i]  # (N, 3)
                F_loss = ((F_pred - F_tar_i)**2).sum() / (3 * N)  # per‑atom MSE
                loss_F_tot += F_loss.item()

            loss_E_avg = loss_E_tot / batch_size
            loss_F_avg = loss_F_tot / batch_size

            # 50–50 energy–forces
            total_loss = 0.5 * loss_E_avg + 0.5 * loss_F_avg
            print("Running batch loss tracker. E_loss:", loss_E_avg," F_loss:", loss_F_avg, " total loss:", total_loss) 
            loss_tensor = torch.tensor(total_loss, device=device, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
            epoch_loss += total_loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        print("Running epoch loss tracker:", avg_epoch_loss, "last epoch's loss:", last_epoch_loss) 
        train_losses.append(avg_epoch_loss)
        last_epoch_loss = avg_epoch_loss

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (node_batch, edge_batch, triplet_batch, pos_batch,
                 E_total_tar, F_tar, N_batch) in test_loader:
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

                    total_E_actual, F_pred = model.energy_and_forces(node_i, edge_i, trip_i, pos_i)
                    F_tar_i = F_tar[i]
                    F_loss = ((F_pred - F_tar_i)**2).sum() / (3 * N)

                    batch_loss += 0.5 * loss_E + 0.5 * F_loss

                test_loss += batch_loss / batch_size

            test_loss /= len(test_loader)
            test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_epoch_loss:.4f}, Test Loss = {test_loss:.4f}")

    model.save('wedge_model.pt')
    return train_losses, test_losses


# VISUALIZATION
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss (50/50 E+F)')
    plt.plot(test_losses,  label='Test Loss  (50/50 E+F)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('WedgeForceField Training (LJ‑H2O/HF, 50/50 energy–forces)')
    plt.legend()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()


# MAIN EXECUTION
def main():
    mol_filename = 'training_set.xyz'
    model_filename = 'wedge_model.pt'

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
    model = WedgeMLP()
    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, map_location=device))
        print(f"Training model loaded from {model_filename}")
    else:
        print("Training new model...")

    lr = 1e-4
    print("Learning rate:", lr)
    optimizer = optim.Adam(model.parameters(), lr)
    train_losses, test_losses = train_model_50_50(model, train_loader, test_loader, optimizer)
    plot_losses(train_losses, test_losses)

    # 5. Single‑molecule test (H2O equilibrium)
    pos_test = np.array([[0.,0.,0.],
                         [0.96,0.,0.],
                         [-0.48,0.83,0.]])
    els = ['O','H','H']
    target_E = lj_potential(pos_test, 0.1)
    target_F = lj_forces(pos_test, 0.1)

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
