# Four-way architecture comparison for a water MLFF
# Optimised for GPU by supervisor, but model frameworks were developed by me
import json
import math
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from ase.io import read

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# configuration 
DATASET_PATH = './water_dataset_64.extxyz'
SPLIT_JSON   = './split.json'      # split saved once, reused by every model/run
BATCH_SIZE   = 8
ACCUM_STEPS  = 4                   # 8 x 4 = 32 effective batch
EPOCHS       = 250
LR           = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 5.0
WARMUP_E     = 20                  # energy-only epochs (wake readout + block)
RAMP_F       = 30                  # epochs to reach full force weight
ES_PATIENCE  = 50                  # early stop on score (counted only after full ramp)
CUTOFF_EDGE  = 6.0                 # pairwise range (attention masked to this too)
CUTOFF_TRIP  = 4.0                 # triplet range (fixed across four models)
N_RBF        = 16
FAN_WIDTH    = 16
D_MODEL      = 64
EPTFF_REF    = 0.40

# element embedding
def _first_n_primes(n):
    primes, cand = [], 2
    while len(primes) < n:
        if all(cand % p for p in primes):
            primes.append(cand)
        cand += 1
    return primes

ROW_PRIMES = _first_n_primes(7)    # periods 1..7
COL_PRIMES = _first_n_primes(18)   # groups 1..18

PERIODIC = {'H': (1, 1), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16)}
LOG_COORDS = {el: (math.log(ROW_PRIMES[p - 1]), math.log(COL_PRIMES[g - 1]))
              for el, (p, g) in PERIODIC.items()}
SUPPORTED_ELEMENTS = list(PERIODIC)

for el, (x, y) in LOG_COORDS.items():
    print(f"  element {el}: coordinate ({x:.4f}, {y:.4f})")

# data loading 

def load_from_extxyz(filename):
    """Load molecular structures and their reference properties from extxyz file

    Parameters:
    -----------
        filename: str
            Path to the extxyz dataset file

    Returns:
    ---------
        molecules: list
            List containing atomic positions, energies, forces, cells and element symbols
    """
    frames = read(filename, index=':')
    molecules = []
    random.shuffle(frames)
    for atoms in frames:
        pos_t = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        E_total = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        F_atom_t = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        elements = [atom.symbol for atom in atoms]
        for el in elements:
            if el not in LOG_COORDS:
                raise ValueError(f"Unsupported element {el}; supported: {SUPPORTED_ELEMENTS}")
        lengths = np.array(atoms.cell.lengths())
        if atoms.pbc.any() and (lengths > 0).all():
            cell = lengths
        else:
            cell = np.full(3, 1e4)
        cell_t = torch.tensor(cell, dtype=torch.float32)
        molecules.append((pos_t, E_total, F_atom_t, cell_t, elements))
    print(f'Loaded {len(molecules)} frames from {filename}')
    return molecules


def molecules_to_tensors(molecules, dev):
    """Convert molecular data into PyTorch tensors

    Parameters:
    --------------
        molecules: list
            List of molecular structures and their reference properties
        dev: torch.device
            Device used to store the tensors

    Returns:
    --------
        samples: list
            List of molecular data stored as PyTorch tensors
        targets_E: list
            List of reference total energies
    """
    samples, targets_E = [], []
    for pos, E_total, F_atom, cell_t, elements in molecules:
        node_t = torch.tensor([LOG_COORDS[el] for el in elements],
                              dtype=torch.float32, device=dev)
        samples.append((node_t, pos.to(dev), E_total, F_atom.to(dev), cell_t.to(dev)))
        targets_E.append(E_total.item())
    return samples, targets_E

# vectorized pairwise / triplet geometry 

def build_graph_features(pos, cell,
                         cutoff_edge=CUTOFF_EDGE, cutoff_trip=CUTOFF_TRIP):
    """Builds pairwise and three-atom graph features from atomic coordinates
        
        Parameters: 
        --------------
    
        pos : torch.Tensor
            Atomic positions
        cell : torch.Tensor
            Simulation cell dimensions
        cutoff_edge : float
            Maximum distance for including an atom pair as an edge
        cutoff_trip : float
            Maximum distance for including atoms in a triplet
    
        Returns:
        -------------
    
        dict
            Dictionary of distances, atomic indices, and masks for edges and triplets
        """
    B, N, _ = pos.shape
    dev = pos.device

    raw = pos.unsqueeze(2) - pos.unsqueeze(1)
    cell4 = cell[:, None, None, :]
    R = raw - cell4 * torch.round(raw / cell4)
    # memory  optimisation
    del raw
    D = R.norm(dim=-1)

    # Triplets (i, c, j): all angles at central atom c 
    A = (D < cutoff_trip) & ~torch.eye(N, dtype=torch.bool, device=dev)
    pairs = torch.triu_indices(N, N, offset=1, device=dev)
    i_p, j_p = pairs[0], pairs[1]

    Tmask = A[:, :, i_p] & A[:, :, j_p]
    # memory  optimisation
    del A
    counts3 = Tmask.sum(dim=(1, 2))
    b_t, c_t, p_t = Tmask.nonzero(as_tuple=False).unbind(1)
    # memory  optimisation
    del Tmask
    i_t, j_t = i_p[p_t], j_p[p_t]

    v1 = R[b_t, i_t, c_t]
    v2 = R[b_t, j_t, c_t]
    d1 = D[b_t, i_t, c_t]
    d2 = D[b_t, j_t, c_t]
    dij = D[b_t, i_t, j_t]
    cos = ((v1 * v2).sum(-1) / (d1 * d2 + 1e-8)).clamp(-1.0, 1.0)

    trip_d = torch.stack([cos, d1, d2, dij], dim=-1)
    trip_atom = torch.stack([i_t, c_t, j_t], dim=-1)

    tmax = max(int(counts3.max()), 1)
    sizes = counts3.tolist()
    trip_d = torch.stack([Fn.pad(t, (0, 0, 0, tmax - t.shape[0]))
                          for t in trip_d.split(sizes)])
    trip_atom = torch.stack([Fn.pad(t, (0, 0, 0, tmax - t.shape[0]))
                             for t in trip_atom.split(sizes)])
    trip_mask = torch.arange(tmax, device=dev).unsqueeze(0) < counts3.unsqueeze(1)

    # Edges with long cutoff 
    d_p = D[:, i_p, j_p]
    emask = d_p < cutoff_edge
    counts_e = emask.sum(dim=1)
    b_e, p_e = emask.nonzero(as_tuple=False).unbind(1)
    del emask
    edge_d = d_p[b_e, p_e]
    edge_atom = torch.stack([i_p[p_e], j_p[p_e]], dim=-1)
    emax = max(int(counts_e.max()), 1)
    sezs = counts_e.tolist()
    edge_d = torch.stack([Fn.pad(t, (0, emax - t.shape[0]))
                          for t in edge_d.split(sezs)])
    edge_atom = torch.stack([Fn.pad(t, (0, 0, 0, emax - t.shape[0]))
                             for t in edge_atom.split(sezs)])
    edge_pid = torch.stack([Fn.pad(t, (0, emax - t.shape[0]))
                            for t in p_e.split(sezs)])
    edge_mask = torch.arange(emax, device=dev).unsqueeze(0) < counts_e.unsqueeze(1)

    return {'edge_d': edge_d, 'edge_atom': edge_atom, 'edge_mask': edge_mask,
            'edge_pid': edge_pid,
            'trip_d': trip_d, 'trip_atom': trip_atom, 'trip_mask': trip_mask}

# graph helpers 
# Pair ordering for edges
def _triu_index(u, v, N):
    """Assigns unique indexes to each pair of atoms to prevent duplicate pairs
        
        Parameters:
        ------------
        u, v : torch.Tensor 
            Indices of two atoms 
        
        N : int
            Unique index representing the pair of atoms """
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    return lo * (2 * N - lo - 1) // 2 + (hi - lo - 1)


def scatter_sum(src, index, B, N):
    """Sum values according to their indices for each batch 
        
        Parameters:
        ----------
        src : torch.Tensor
            Values to be summed
        index : torch.Tensor
            Indices specifying where each value should be added
        B : int
            Number of batches
        N : int
            Number of nodes or atoms per batch
    
        Returns
        -------
        torch.Tensor
            Tensor containing the summed values for each node in each batch
        """
    b = torch.arange(B, device=src.device).unsqueeze(1).expand(B, index.shape[1])
    out = src.new_zeros(B * N, src.shape[-1])
    out.index_add_(0, (b * N + index).reshape(-1), src.reshape(-1, src.shape[-1]))
    return out.view(B, N, -1)


def scatter_mean(src, index, weight, B, N):
    """Calculates a weighted mean of values for each node
        
        Parameters:
        ----------
        src: torch.Tensor
            Values or features to be averaged
        index : torch.Tensor
            Indices specifying which node each value belongs to
        weight : torch.Tensor
            Weight applied to each value
        B : int
            Number of batches
        N : int
            Number of nodes or atoms in each batch
    
        Returns
        -------
        torch.Tensor
            Weighted mean values for each node"""
    w = weight.unsqueeze(-1)
    num = scatter_sum(src * w, index, B, N) 
    den = scatter_sum(w, index, B, N)       
    return num / den.clamp(min=1e-3)


def gather_nodes(h, idx):
    """Retrieve node features using specified indices
    
        Parameters
        ----------
        h : torch.Tensor
            Node feature tensor
        idx : torch.Tensor
            Indices of the nodes to retrieve
    
        Returns
        -------
        torch.Tensor
            Features of the selected nodes
        """
    return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

# basis / cutoff 

def smooth_cutoff(r, rc):
    """Applies a smooth cosine cutoff to atomic distances
        Parameters: 
        ---------------
            r: torch.Tensor
                Distances between atoms 
            rc: float
                Cutoff distance 
        
        Returns:
        ---------------
        torch.Tensor
            Smooth cut off values from 1 to 0.
        """
    x = (r / rc).clamp(max=1.0)
    s = 0.5 * (1.0 + torch.cos(math.pi * x))
    return torch.where(r < rc, s, torch.zeros_like(s))


class RadialBasis(nn.Module):
    """Convert distancs into radial basis function features 
            
            Parameters:
            -----------
            r: torch.Tensor
                Interatomic distances 
                
            Returns:
            ---------
            torch.Tensor 
                Gaussian radial distribution features for each distance"""
    def __init__(self, r_min, r_max, n_basis):
        super().__init__()
        self.register_buffer('centers', torch.linspace(r_min, r_max, n_basis))
        self.sigma = (r_max - r_min) / max(n_basis - 1, 1)

    def forward(self, r):
        return torch.exp(-((r.unsqueeze(-1) - self.centers) ** 2)
                         / (2 * self.sigma ** 2))

# frame-free invariant edge tokens 

class EdgeTokens(nn.Module):
    """Create feature tokens for edges using distances and triplet information
    
        Parameters
        ----------
        rbf_edge : nn.Module
            Radial basis function used to encode edge distances
        rbf_trip : nn.Module
            Radial basis function used to encode triplet distances
        fan_width : int, optional
            Size of the triplet interaction features
        cutoff_edge : float, optional
            Maximum distance used for edge interactions
        cutoff_trip : float, optional
            Maximum distance used for triplet interactions
        """
    def __init__(self, rbf_edge, rbf_trip, fan_width=FAN_WIDTH):
        super().__init__()
        self.rbf_edge, self.rbf_trip = rbf_edge, rbf_trip
        self.fan_width = fan_width
        n_rbf = rbf_trip.centers.numel()
        self.psi = nn.Sequential(nn.Linear(1 + n_rbf, fan_width), nn.Mish(),
                                 nn.Linear(fan_width, fan_width))

    def forward(self, feats, N):
        """
                Generate edge tokens from graph features
        
                Parameters:
                ----------
                feats : dict
                    Graph features containing edge distances, triplet distances,
                    atom indices, and masks
                N : int
                    Number of atoms in each structure
        
                Returns:
                -------
                tuple
                    Edge feature tokens and triplet interaction features
                """
        d = feats['edge_d']                                    # (B,E)
        base = torch.cat([torch.ones_like(d).unsqueeze(-1),    # affine carrier
                          (d * d / (CUTOFF_EDGE * CUTOFF_EDGE)).unsqueeze(-1),
                          self.rbf_edge(d)], dim=-1)           # (B,E,2+n_rbf)

        B, E = d.shape
        P = N * (N - 1) // 2
        pid_map = feats['edge_pid'].new_zeros((B, P))
        em = feats['edge_mask']
        bidx = torch.arange(B, device=d.device).unsqueeze(1).expand(B, E)
        eids = torch.arange(E, device=d.device).unsqueeze(0).expand(B, E) + 1
        pid_map[bidx[em], feats['edge_pid'][em]] = eids[em]

        a, c, b = feats['trip_atom'].unbind(-1)
        cos, d1, d2, _ = feats['trip_d'].unbind(-1)
        psi = self.psi(torch.cat([cos.unsqueeze(-1),
                                  self.rbf_trip(d1) + self.rbf_trip(d2)], -1))
        psi = psi * (smooth_cutoff(d1, CUTOFF_TRIP)
                     * smooth_cutoff(d2, CUTOFF_TRIP)).unsqueeze(-1)
        psi = psi * feats['trip_mask'].unsqueeze(-1)           # pads -> exactly 0

        fan = d.new_zeros(B, E, self.fan_width)
        for leg in (a, b):
            tid = _triu_index(c, leg, N).clamp(min=0)          # pad triplets clamp
            e0 = pid_map.gather(1, tid)                        # 0 if missing
            eid = (e0 - 1).clamp(min=0)                        # real legs ALWAYS found:
                                                               # trip cut 4 <= edge cut 6
            fan.scatter_add_(1, eid.unsqueeze(-1).expand(-1, -1, self.fan_width), psi)
        token = torch.cat([base, fan], -1)                     # (B,E,2+n_rbf+fan)
        return token, psi

# the four context blocks 
def _small_init_last(seq, std=1e-2):
    """Initialise the final layer with small random weights
    
        Parameters:
            seq : nn.Sequential
                Neural network sequence containing the final layer
            std : float
                Standard deviation used for weight initialisation
    
        Returns:
            None
                Initialises the weights and bias of the final layer in place
        """
    nn.init.normal_(seq[-1].weight, std=std)
    nn.init.zeros_(seq[-1].bias)


class MlpBlock(nn.Module):
    """Neural network block for updating atom features using edge information
    
        Parameters:
            d : int
                Dimension of the atom feature vectors
            token_dim : int
                Dimension of the edge token features
            n_rbf : int
                Number of radial basis function features
            width : int
                Width of the hidden layer
    
        Returns:
            None
                Initialises the neural network layers
        """
    def __init__(self, d, token_dim, n_rbf, width=96):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d + token_dim, width), nn.Mish(),
                                 nn.Linear(width, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.Mish(), nn.Linear(d, d))
        _small_init_last(self.upd)

    def forward(self, h, feats, token, psi):
        """Update atom features using neighbouring atom information
        
                Parameters:
                    h : torch.Tensor
                        Atom feature tensor
                    feats : dict
                        Graph features containing edge distances and atom indices
                    token : torch.Tensor
                        Edge feature tokens
                    psi : torch.Tensor
                        Triplet interaction features
    
                Returns:
                    torch.Tensor
                        Updated atom feature tensor
                """
        B, N, _ = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], CUTOFF_EDGE) * feats['edge_mask']
        m = self.msg(torch.cat([gather_nodes(h, j), token], -1))
        agg = scatter_mean(m, i, keep, B, N)
        return self.upd(torch.cat([h, agg], -1))


class MhaBlock(nn.Module):
    """Multi-head attention block 
    
        Parameters:
            d : int
                Dimension of the atom feature vectors
            token_dim : int
                Dimension of the edge token features
            n_rbf : int
                Number of radial basis function features
            heads : int
                Number of attention heads
    
        Returns:
            None
                Initialises the attention layers
        """
    def __init__(self, d, token_dim, n_rbf, heads=4):
        super().__init__()
        assert d % heads == 0
        self.hd, self.dk, self.d = heads, d // heads, d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.bias = nn.Linear(token_dim, heads)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.Mish(), nn.Linear(d, d))
        _small_init_last(self.proj)

    def forward(self, h, feats, token, psi):
        B, N, D = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], CUTOFF_EDGE) * feats['edge_mask']
        qh = self.q(gather_nodes(h, i)).view(B, -1, self.hd, self.dk)
        kh = self.k(gather_nodes(h, j)).view(B, -1, self.hd, self.dk)
        logit = (qh * kh).sum(-1) / math.sqrt(self.dk) + self.bias(token)
        logit = logit.masked_fill(~feats['edge_mask'].unsqueeze(-1), float('-inf'))
        a = logit.exp()                                        # pads -> exactly 0
        den = scatter_sum(a, i, B, N)                          # (B,N,heads)
        alpha = a / (den.gather(1, i.unsqueeze(-1).expand(-1, -1, self.hd)) + 1e-9)
        vh = self.v(gather_nodes(h, j)).view(B, -1, self.hd, self.dk)
        av = (alpha.unsqueeze(-1) * vh) * keep.unsqueeze(-1).unsqueeze(-1)
        return self.proj(scatter_sum(av.reshape(B, -1, D), i, B, N))


class ConvBlock(nn.Module):
    """Convolutional block for combining pairwise and triplet information
    
        Parameters:
            d : int
                Dimension of the atom feature vectors
            token_dim : int
                Dimension of the edge token features
            n_rbf : int
                Number of radial basis function features
            fan_width : int
                Size of the triplet interaction features
            ang_width : int
                Width of the hidden angular interaction layer
    
        Returns:
            None
                Initialises the convolutional layers
        """
    def __init__(self, d, token_dim, n_rbf, fan_width=FAN_WIDTH, ang_width=48):
        super().__init__()
        self.n_rbf, self.fan_width = n_rbf, fan_width
        self.pair = nn.Sequential(nn.Linear(2 * d + token_dim, d), nn.Mish(),
                                  nn.Linear(d, d))
        self.gate = nn.Linear(n_rbf, d)
        self.ang = nn.Sequential(nn.Linear(3 * d + fan_width, ang_width), nn.Mish(),
                                 nn.Linear(ang_width, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.Mish(), nn.Linear(d, d))
        _small_init_last(self.fuse)

    def forward(self, h, feats, token, psi):
        B, N, _ = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], CUTOFF_EDGE) * feats['edge_mask']
        hi, hj = gather_nodes(h, i), gather_nodes(h, j)
        mp = self.pair(torch.cat([hi, hj, token], -1))
        g = torch.sigmoid(self.gate(token[..., 2:2 + self.n_rbf]))
        pair_agg = scatter_mean(mp * g, i, keep, B, N) \
                 + scatter_mean(mp * g, j, keep, B, N)

        it, c, jt = feats['trip_atom'].unbind(-1)
        d1, d2 = feats['trip_d'][..., 1], feats['trip_d'][..., 2]
        lenv = (smooth_cutoff(d1, CUTOFF_TRIP)
                * smooth_cutoff(d2, CUTOFF_TRIP)) * feats['trip_mask']
        mt = self.ang(torch.cat([gather_nodes(h, it), gather_nodes(h, jt),
                                 gather_nodes(h, c), psi], -1)) * lenv.unsqueeze(-1)
        trip_agg = scatter_mean(mt, c, feats['trip_mask'].float(), B, N)
        return self.fuse(torch.cat([pair_agg, trip_agg], -1))


class CombinedBlock(nn.Module):
    """Combined black box model that uses attention, convolutional, angular and global features
         Parameters:
            d : int
                Dimension of the atom feature vectors
            token_dim : int
                Dimension of the edge token features
            n_rbf : int
                Number of radial basis function features
            fan_width : int
                Size of the triplet interaction features
            heads : int
                Number of attention heads
            ang_width : int
                Width of the hidden angular interaction layer
    
        Returns:
            None
                Initialises the combined neural network layers
        """
    def __init__(self, d, token_dim, n_rbf, fan_width=FAN_WIDTH, heads=4, ang_width=48):
        super().__init__()
        self.hd, self.dk, self.d = heads, d // heads, d
        self.n_rbf = n_rbf
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.att_bias = nn.Linear(token_dim, heads)
        self.gate = nn.Linear(n_rbf, d)
        self.ang = nn.Sequential(nn.Linear(3 * d + fan_width, ang_width), nn.Mish(),
                                 nn.Linear(ang_width, d))
        self.glob = nn.Sequential(nn.Linear(d, d), nn.Mish(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(4 * d, d), nn.Mish(), nn.Linear(d, d))
        _small_init_last(self.fuse)

    def forward(self, h, feats, token, psi):
        B, N, D = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], CUTOFF_EDGE) * feats['edge_mask']
        hi, hj = gather_nodes(h, i), gather_nodes(h, j)
        qh = self.q(hi).view(B, -1, self.hd, self.dk)
        kh = self.k(hj).view(B, -1, self.hd, self.dk)
        logit = (qh * kh).sum(-1) / math.sqrt(self.dk) + self.att_bias(token)
        logit = logit.masked_fill(~feats['edge_mask'].unsqueeze(-1), float('-inf'))
        a = logit.exp()                                        # pads -> exactly 0
        den = scatter_sum(a, i, B, N)
        alpha = a / (den.gather(1, i.unsqueeze(-1).expand(-1, -1, self.hd)) + 1e-9)
        vh = self.v(hj).view(B, -1, self.hd, self.dk)
        av = (alpha.unsqueeze(-1) * vh) * keep.unsqueeze(-1).unsqueeze(-1)
        attn = scatter_sum(av.reshape(B, -1, D), i, B, N)
        g = torch.sigmoid(self.gate(token[..., 2:2 + self.n_rbf]))
        conv = scatter_mean(g * hj, i, keep, B, N) \
             + scatter_mean(g * hj, j, keep, B, N)
        it, c, jt = feats['trip_atom'].unbind(-1)
        d1, d2 = feats['trip_d'][..., 1], feats['trip_d'][..., 2]
        lenv = (smooth_cutoff(d1, CUTOFF_TRIP)
                * smooth_cutoff(d2, CUTOFF_TRIP)) * feats['trip_mask']
        mt = self.ang(torch.cat([gather_nodes(h, it), gather_nodes(h, jt),
                                 gather_nodes(h, c), psi], -1)) * lenv.unsqueeze(-1)
        ang = scatter_mean(mt, c, feats['trip_mask'].float(), B, N)
        glob = self.glob(h.mean(dim=1, keepdim=True)).expand(B, N, D)
        return self.fuse(torch.cat([attn, conv, ang, glob], -1))

MODELS = {
    'mlp':  (MlpBlock,  {}),
    'mha':  (MhaBlock,  {'heads': 4}),
    'conv': (ConvBlock, {}),
    'comb': (CombinedBlock, {'heads': 4}),
}

# model wrapper
# One complete potential = shared embedder + ONE context block + per-atom

class ForceField(nn.Module):
    def __init__(self, block_cls, d=D_MODEL, n_rbf=N_RBF, **block_kw):
        """Initialise the machine learning force field

        Parameters:
            block_cls: type
                Context block class used by the force field
            d: int
                Dimension of the atom feature representation
            n_rbf: int
                Number of radial basis functions
            block_kw: dict
                Additional arguments passed to the context block
        """
        super().__init__()
        self.rbf_edge = RadialBasis(0.4, CUTOFF_EDGE, n_rbf)
        self.rbf_trip = RadialBasis(0.4, CUTOFF_TRIP, n_rbf)
        token_dim = 2 + n_rbf + FAN_WIDTH
        self.edge_tokens = EdgeTokens(self.rbf_edge, self.rbf_trip)
        self.embed = nn.Sequential(nn.Linear(2, d), nn.Mish(), nn.Linear(d, d))
        self.one_body = nn.Linear(2, 1)
        self.block = block_cls(d, token_dim, n_rbf, **block_kw)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.Mish(), nn.Linear(d, 1))
        nn.init.normal_(self.readout[-1].weight, std=1e-2)   # NOT zero (see docstring)
        nn.init.zeros_(self.readout[-1].bias)
        nn.init.zeros_(self.one_body.weight)
        nn.init.zeros_(self.one_body.bias)

    def forward(self, node, pos, cell):
        feats = build_graph_features(pos, cell) 
        token, psi = self.edge_tokens(feats, pos.shape[1])
        h = self.embed(node)
        if self.training and USE_CHECKPOINT:
            dh = torch.utils.checkpoint.checkpoint(
                self.block, h, feats, token, psi, use_reentrant=False)
        else:
            dh = self.block(h, feats, token, psi)
        h = h + dh
        e_i = self.readout(h).squeeze(-1) + self.one_body(node).squeeze(-1)
        return e_i.sum(-1) 

#  data pipeline

def collate_fn(batch):
    """Combine individual samples into a batch for model training

    Parameters:
        batch: list
            List of individual molecular samples

    Returns:
        dict
            Batched molecular features and target properties
    """
    return {'node': torch.stack([s['node'] for s in batch]),
            'pos': torch.stack([s['pos'] for s in batch]),
            'E_target': torch.stack([s['E_target'] for s in batch]),
            'F_target': torch.stack([s['F_target'] for s in batch]),
            'cell': torch.stack([s['cell'] for s in batch])}


class EnergyDataset(Dataset):
    def __init__(self, samples):
        """Initialise the dataset

        Parameters:
            samples: list
                List of molecular samples
        """
        self.samples = samples

    def __len__(self):
        """Return the number of molecular samples

        Returns:
            int
                Number of samples in the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Return one molecular sample

        Parameters:
            idx: int
                Index of the sample to retrieve

        Returns:
            dict
                Molecular features and target energy and forces
        """
        node_t, pos_t, E_total, F_atom_t, cell_t = self.samples[idx]
        return {'node': node_t, 'pos': pos_t, 'E_target': E_total,
                'F_target': F_atom_t, 'cell': cell_t}


# Split once and lee[ identical data across models and reruns.
def make_split(n):
    """Create or load fixed training, validation and test splits

    Parameters:
        n: int
            Total number of samples in the dataset

    Returns:
        train: list
            Indices of the training samples
        val: list
            Indices of the validation samples
        test: list
            Indices of the test samples
    """
    if os.path.exists(SPLIT_JSON):
        s = json.load(open(SPLIT_JSON))
        if s.get('n') == n:
            print(f"Loaded fixed split from {SPLIT_JSON}")
            return s['train'], s['val'], s['test']
    idx = torch.randperm(n).tolist()
    s = {'n': n,
         'train': idx[:int(0.8 * n)],
         'val': idx[int(0.8 * n):int(0.9 * n)],
         'test': idx[int(0.9 * n):]}
    json.dump(s, open(SPLIT_JSON, 'w'))
    print(f"Saved fixed split to {SPLIT_JSON}")
    return s['train'], s['val'], s['test']


def compute_predictions(model, batch, need_forces, create_graph=False):
    """Calculate predicted energies and derive forces

    Parameters:
        model: torch.nn.Module
            Machine learning force field used for prediction
        batch: dict
            Batch of molecular features and target properties
        need_forces: bool
            Whether forces should be calculated from the energy gradient
        create_graph: bool
            Whether to keep the computational graph for higher-order gradients

    Returns:
        E_pred: torch.Tensor
            Predicted total energies
        f_pred: torch.Tensor or None
            Predicted atomic forces or None if forces are not calculated
        E_tar: torch.Tensor
            Reference total energies
        F_tar: torch.Tensor
            Reference atomic forces
    """
    node = batch['node'].to(device)
    pos = batch['pos'].to(device)
    if need_forces:
        pos.requires_grad_(True)
    cell = batch['cell'].to(device)
    E_tar = batch['E_target'].to(device)
    F_tar = batch['F_target'].to(device)

    E_pred = model(node, pos, cell)
    f_pred = None
    if need_forces:
        f_pred = -torch.autograd.grad(E_pred.sum(), pos, create_graph=create_graph)[0]
    return E_pred, f_pred, E_tar, F_tar

#  metrics 

def element_ids(node):
    codes = torch.tensor([LOG_COORDS[el] for el in PERIODIC],
                         device=node.device, dtype=node.dtype)
    return (node.unsqueeze(-2) - codes).norm(dim=-1).argmin(-1)


def accumulate_force_stats(stats, f_pred, f_tar, node):
    diff = (f_pred - f_tar).reshape(-1)
    tgt = f_tar.reshape(-1)
    stats['sse'] += (diff ** 2).sum().item()
    stats['fsq'] += (tgt ** 2).sum().item()
    stats['fsum'] += tgt.sum().item()
    stats['n'] += tgt.numel()
    ids = element_ids(node)
    for k, el in enumerate(PERIODIC):
        m = ids == k
        if m.any():
            stats['el_se'][el] += ((f_pred - f_tar)[m] ** 2).sum().item()
            stats['el_n'][el] += int(m.sum())


def finalize_force_stats(stats):
    out = {'r2': 1.0 - stats['sse'] / (stats['fsq'] - stats['fsum'] ** 2 / stats['n'])}
    out['el'] = {el: stats['el_se'][el] / stats['el_n'][el]
                 for el in PERIODIC if stats['el_n'][el] > 0}
    return out
# --------------------------------- training ----------------------------------

def _train(name, cls, kw, dataset, mean_E, varE, varF, train_loader, val_loader):
    """Train one force field architecture and save its best checkpoint

    Parameters:
        name: str
            Name of the model architecture
        cls: type
            Context block class used by the model
        kw: dict
            Additional arguments for the context block
        dataset: EnergyDataset
            Complete molecular dataset
        mean_E: float
            Mean reference energy used to shift the target energies
        varE: float
            Variance of the reference energies used to normalise energy loss
        varF: float
            Mean squared reference force used to normalise force loss
        train_loader: DataLoader
            DataLoader containing the training data
        val_loader: DataLoader
            DataLoader containing the validation data

    Returns:
        dict
            Training, number of parameters and best validation results"""
    model = ForceField(cls, **kw).to(device)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                               patience=10, min_lr=1e-6)
    hist = {'trE': [], 'valE': [], 'trF': [], 'valF': []}
    best_s, best_f, best_e, bad = float('inf'), float('inf'), float('inf'), 0
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        w = 0.0 if epoch < WARMUP_E else min(1.0, (epoch - WARMUP_E) / RAMP_F)
        model.train()
        optimizer.zero_grad()
        e_sum, f_sum = 0.0, 0.0

        for step, batch in enumerate(train_loader):
            try:
                # warmup: NO force graph at all (no autograd.grad call) — the
                E_pred, f_pred, E_tar, f_tar = compute_predictions(
                    model, batch, need_forces=(w > 0), create_graph=True)
                loss_E = Fn.mse_loss(E_pred, E_tar)
                loss = loss_E / varE / ACCUM_STEPS
                loss_F = None
                if w > 0:
                    loss_F = Fn.mse_loss(f_pred, f_tar)
                    loss = loss + w * loss_F / varF / ACCUM_STEPS
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"  non-finite loss at epoch {epoch+1}, step {step} — "
                          f"batch skipped")
                    continue
                loss.backward()
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  OOM at epoch {epoch+1}, step {step} — batch skipped")
                continue

            e_sum += loss_E.item()
            if loss_F is not None:
                f_sum += loss_F.item()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=GRAD_CLIP)
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    print(f"  non-finite grad norm at epoch {epoch+1}, "
                          f"step {step} — update skipped")
                optimizer.zero_grad()

        # validation (Forces need a first-order graph, so still has grad)
        model.eval()
        vE, vF = 0.0, 0.0
        stats = {'sse': 0.0, 'fsq': 0.0, 'fsum': 0.0, 'n': 0,
                 'el_se': {el: 0.0 for el in PERIODIC},
                 'el_n': {el: 0 for el in PERIODIC}}
        for batch in val_loader:
            E_pred, f_pred, E_tar, f_tar = compute_predictions(
                model, batch, need_forces=True, create_graph=False)
            vE += Fn.mse_loss(E_pred, E_tar).item()
            vF += Fn.mse_loss(f_pred, f_tar).item()
            accumulate_force_stats(stats, f_pred, f_tar, batch['node'].to(device))
        nb, nv = len(train_loader), len(val_loader)
        vEm, vFm = vE / nv, vF / nv
        hist['trE'].append(e_sum / nb)
        hist['valE'].append(vEm)
        hist['trF'].append(f_sum / nb if w > 0 else float('nan')) 
        hist['valF'].append(vFm)
        score = vEm / varE + vFm / varF
        scheduler.step(score)

        if score < best_s - 1e-6:
            best_s, best_f, best_e, bad = score, vFm, vEm, 0
            torch.save({'state_dict': model.state_dict(), 'block': name,
                        'mean_E': mean_E, 'cutoff_edge': CUTOFF_EDGE,
                        'cutoff_trip': CUTOFF_TRIP, 'periodic': PERIODIC,
                        'row_primes': ROW_PRIMES, 'col_primes': COL_PRIMES,
                        'n_params': n_params, 'val_F': best_f, 'val_E': best_e,
                        'val_score': best_s, 'epoch': epoch + 1,
                        'history': {k: list(v) for k, v in hist.items()}},
                       f'best_{name}.pt')
        elif w >= 1.0:
            bad += 1

        if (epoch + 1) % 10 == 0:
            fs = finalize_force_stats(stats)
            el_txt = "  ".join(f"{el} {v:.3f}" for el, v in fs['el'].items())
            print(f"  [{name}] Ep {epoch+1:3d} (fw {w:.2f}) | trE {hist['trE'][-1]:8.4f} "
                  f"valE {vEm:8.4f} | trF {hist['trF'][-1]:.6f} "
                  f"valF {vFm:.6f} | score {score:6.3f} | R2 {fs['r2']:.3f} | {el_txt}")
        if bad >= ES_PATIENCE:
            print(f"  [{name}] early stop at epoch {epoch+1} (score patience {ES_PATIENCE})")
            break

    dt = (time.perf_counter() - t0) / (epoch + 1)
    print(f"  [{name}] done: best score {best_s:.4f} (valF {best_f:.6f}, "
          f"valE {best_e:.4f}), {n_params} params, {dt:.1f} s/epoch")
    return {'history': hist, 'n_params': n_params,
            'best_valF': best_f, 'best_valE': best_e, 'best_score': best_s}


def evaluate_test(name, dataset, test_idx):
    """Touch the test set exactly ONCE, on the best-score checkpoint."""
    ckpt = torch.load(f'best_{name}.pt', map_location=device, weights_only=False)
    cls, kw = MODELS[name]
    model = ForceField(cls, **kw).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE,
                        collate_fn=collate_fn)
    tE, tF = 0.0, 0.0
    stats = {'sse': 0.0, 'fsq': 0.0, 'fsum': 0.0, 'n': 0,
             'el_se': {el: 0.0 for el in PERIODIC},
             'el_n': {el: 0 for el in PERIODIC}}
    for batch in loader:
        E_pred, f_pred, E_tar, f_tar = compute_predictions(
            model, batch, need_forces=True, create_graph=False)
        tE += Fn.mse_loss(E_pred, E_tar).item()
        tF += Fn.mse_loss(f_pred, f_tar).item()
        accumulate_force_stats(stats, f_pred, f_tar, batch['node'].to(device))
    st = finalize_force_stats(stats)
    return {'testE': tE / len(loader), 'testF': tF / len(loader),
            'r2': st['r2'], 'el': st['el']}

# main
def main():
    print("Loading training data...")
    molecules = load_from_extxyz(DATASET_PATH)
    samples, targets_E = molecules_to_tensors(molecules, device)

    mean_E = np.mean(targets_E)
    print(f"Shifting energies by {mean_E:.2f} eV")
    samples = [(s[0], s[1], s[2] - mean_E, s[3], s[4]) for s in samples]

    varE = float(np.var(targets_E))
    varF = (torch.cat([s[3] for s in samples]) ** 2).mean().item()
    print(f"zero-force baseline MSE: {varF:.4f} (valF must drop well below this)")
    print(f"energy constant-predictor floor: {varE:.4f} "
          f"(valE pinned here means the model outputs the mean)")
    print(f"loss normalisers: varE {varE:.4f}  varF {varF:.4f}  "
          f"(each loss term starts at 1.0)")

    dataset = EnergyDataset(samples)
    train_idx, val_idx, test_idx = make_split(len(samples))

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                              sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE,
                            collate_fn=collate_fn)

    print("\nParameter counts:")
    for mname, (mcls, mkw) in MODELS.items():
        torch.manual_seed(0)
        m = ForceField(mcls, **mkw)
        print(f"  {mname:5s}: {sum(p.numel() for p in m.parameters()):7d}")
        del m

    results = {}
    for name, (cls, kw) in MODELS.items():
        print(f"\n=== Training model: {name} ({cls.__name__}) ===")
        results[name] = _train(name, cls, kw, dataset, mean_E, varE, varF,
                               train_loader, val_loader)
        results[name]['test'] = evaluate_test(name, dataset, test_idx)

    # summary
    print("\nSUMMARY (test set, touched once)")
    header = (f"{'model':6s} {'params':>8s} {'best valF':>10s} {'best valE':>10s} "
              f"{'testF':>9s} {'testE':>9s} {'R2':>6s}  per-element F MSE")
    print(header)
    for name, r in results.items():
        el_txt = "  ".join(f"{el} {v:.3f}" for el, v in r['test']['el'].items())
        print(f"{name:6s} {r['n_params']:8d} {r['best_valF']:10.6f} "
              f"{r['best_valE']:10.4f} "
              f"{r['test']['testF']:9.6f} {r['test']['testE']:9.4f} "
              f"{r['test']['r2']:6.3f}  {el_txt}")

    print("\nLadder reading:")
    print("  MLP        floor: learned messages, fixed aggregation weights")
    print("  MHA - MLP  value of content-adaptive (softmax) neighbour weighting")
    print("  Conv - MLP value of radial gating + explicit triplet bookkeeping")
    print("  Comb - max(singles)  do the ideas compose, or are they redundant?")

    # comparison plot 
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for name, r in results.items():
        plt.plot(r['history']['valE'], label=name)
    plt.axhline(varE, color='gray', ls=':', label='const-E floor')
    plt.title('Energy MSE (val)'); plt.xlabel('Epoch'); plt.legend()

    plt.subplot(1, 2, 2)
    for name, r in results.items():
        plt.semilogy(r['history']['valF'], label=name)
    plt.axhline(varF, color='gray', ls=':', label='zero-model baseline')
    plt.axhline(EPTFF_REF, color='k', ls='--', label='EPTFF (prior arch)')
    plt.title('Force MSE (val, log)'); plt.xlabel('Epoch'); plt.legend()

    plt.tight_layout()
    plt.savefig('comparison_curves.png', dpi=150)
    plt.show()
    print("Saved comparison_curves.png and best_{mlp,mha,conv,comb}.pt")


if __name__ == "__main__":
    main()
