"""
Unified Comprehensive Testing Framework for MLFF Models vs GFN2-XtB Ground Truth
Including comparison with MACE 

Usage:
    Install ase, tblite, and MACE to a Python environment. 
    To run, use: python test_vs_xtb_ground_truth.py
"""

import json
import math
import os
import random
import time
import warnings
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from ase import Atoms
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.calculators.calculator import Calculator, all_changes
from tblite.ase import TBLite as TBLiteASECalc
from mace.calculators import MACECalculator

import torch
import torch.nn as nn
import torch.nn.functional as F

os.makedirs("test_results/plots", exist_ok=True)
os.makedirs("test_results/csvs", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Element embedding with prime pair.

def _first_n_primes(n):
    """Results a list containing the first n prime numbers
    
    Parameters: 
    ------------
    n : int
        Number of prime numbers to generate
        
    Returns:
    -----------
    primes : list
        List containing the first prime numbers """
    primes, cand = [], 2
    while len(primes) < n:
        if all(cand % p for p in primes):
            primes.append(cand)
        cand += 1
    return primes

ROW_PRIMES = _first_n_primes(7)
COL_PRIMES = _first_n_primes(18)

PERIODIC = {'H': (1, 1), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16)}
LOG_COORDS = {el: (math.log(ROW_PRIMES[p - 1]), math.log(COL_PRIMES[g - 1]))
              for el, (p, g) in PERIODIC.items()}

# models from train_all_models.py
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
    """Represents interatomic distances using Gaussian radial basis functions to 
    represent how interactions change with distance 
    """
    def __init__(self, r_min, r_max, n_basis):
        super().__init__()
        self.register_buffer('centers', torch.linspace(r_min, r_max, n_basis))
        self.sigma = (r_max - r_min) / max(n_basis - 1, 1)

    def forward(self, r):
        """Convert distancs into radial basis function features 
        
        Parameters:
        -----------
        r: torch.Tensor
            Interatomic distances 
            
        Returns:
        ---------
        torch.Tensor 
            Gaussian radial distribution features for each distance"""
        return torch.exp(-((r.unsqueeze(-1) - self.centers) ** 2)
                         / (2 * self.sigma ** 2))


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


def build_graph_features(pos, cell, cutoff_edge=6.0, cutoff_trip=4.0):
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
    del raw
    D = R.norm(dim=-1)

    A = (D < cutoff_trip) & ~torch.eye(N, dtype=torch.bool, device=dev)
    pairs = torch.triu_indices(N, N, offset=1, device=dev)
    i_p, j_p = pairs[0], pairs[1]

    Tmask = A[:, :, i_p] & A[:, :, j_p]
    del A
    counts3 = Tmask.sum(dim=(1, 2))
    b_t, c_t, p_t = Tmask.nonzero(as_tuple=False).unbind(1)
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
    trip_d = torch.stack([F.pad(t, (0, 0, 0, tmax - t.shape[0]))
                          for t in trip_d.split(sizes)])
    trip_atom = torch.stack([F.pad(t, (0, 0, 0, tmax - t.shape[0]))
                             for t in trip_atom.split(sizes)])
    trip_mask = torch.arange(tmax, device=dev).unsqueeze(0) < counts3.unsqueeze(1)

    d_p = D[:, i_p, j_p]
    emask = d_p < cutoff_edge
    counts_e = emask.sum(dim=1)
    b_e, p_e = emask.nonzero(as_tuple=False).unbind(1)
    del emask
    edge_d = d_p[b_e, p_e]
    edge_atom = torch.stack([i_p[p_e], j_p[p_e]], dim=-1)
    emax = max(int(counts_e.max()), 1)
    sezs = counts_e.tolist()
    edge_d = torch.stack([F.pad(t, (0, emax - t.shape[0]))
                          for t in edge_d.split(sezs)])
    edge_atom = torch.stack([F.pad(t, (0, 0, 0, emax - t.shape[0]))
                             for t in edge_atom.split(sezs)])
    edge_pid = torch.stack([F.pad(t, (0, emax - t.shape[0]))
                            for t in p_e.split(sezs)])
    edge_mask = torch.arange(emax, device=dev).unsqueeze(0) < counts_e.unsqueeze(1)

    return {'edge_d': edge_d, 'edge_atom': edge_atom, 'edge_mask': edge_mask,
            'edge_pid': edge_pid,
            'trip_d': trip_d, 'trip_atom': trip_atom, 'trip_mask': trip_mask}


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
    def __init__(self, rbf_edge, rbf_trip, fan_width=16, cutoff_edge=6.0, cutoff_trip=4.0):
        super().__init__()
        self.rbf_edge, self.rbf_trip = rbf_edge, rbf_trip
        self.fan_width = fan_width
        self.cutoff_edge = cutoff_edge
        self.cutoff_trip = cutoff_trip
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
        d = feats['edge_d']
        base = torch.cat([torch.ones_like(d).unsqueeze(-1),
                          (d * d / (self.cutoff_edge * self.cutoff_edge)).unsqueeze(-1),
                          self.rbf_edge(d)], dim=-1)

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
        psi = psi * (smooth_cutoff(d1, self.cutoff_trip)
                     * smooth_cutoff(d2, self.cutoff_trip)).unsqueeze(-1)
        psi = psi * feats['trip_mask'].unsqueeze(-1)

        fan = d.new_zeros(B, E, self.fan_width)
        for leg in (a, b):
            tid = _triu_index(c, leg, N).clamp(min=0)
            e0 = pid_map.gather(1, tid)
            eid = (e0 - 1).clamp(min=0)
            fan.scatter_add_(1, eid.unsqueeze(-1).expand(-1, -1, self.fan_width), psi)
        token = torch.cat([base, fan], -1)
        return token, psi


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

    def forward(self, h, feats, token, psi, cutoff_edge=6.0, cutoff_trip=4.0):
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
            cutoff_edge : float
                Maximum distance for edge interactions
            cutoff_trip : float
                Maximum distance for triplet interactions

        Returns:
            torch.Tensor
                Updated atom feature tensor
        """
        B, N, _ = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], cutoff_edge) * feats['edge_mask']
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

    def forward(self, h, feats, token, psi, cutoff_edge=6.0, cutoff_trip=4.0):
        """Update atom features using multi-head attention

        Parameters:
            h : torch.Tensor
                Atom feature tensor
            feats : dict
                Graph features containing edge distances, atom indices and masks
            token : torch.Tensor
                Edge feature tokens
            psi : torch.Tensor
                Triplet interaction features
            cutoff_edge : float
                Maximum distance for edge interactions
            cutoff_trip : float
                Maximum distance for triplet interactions

        Returns:
            torch.Tensor
                Updated atom feature tensor
        """
        B, N, D = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], cutoff_edge) * feats['edge_mask']
        qh = self.q(gather_nodes(h, i)).view(B, -1, self.hd, self.dk)
        kh = self.k(gather_nodes(h, j)).view(B, -1, self.hd, self.dk)
        logit = (qh * kh).sum(-1) / math.sqrt(self.dk) + self.bias(token)
        logit = logit.masked_fill(~feats['edge_mask'].unsqueeze(-1), float('-inf'))
        a = logit.exp()
        den = scatter_sum(a, i, B, N)
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
    def __init__(self, d, token_dim, n_rbf, fan_width=16, ang_width=48):
        super().__init__()
        self.n_rbf, self.fan_width = n_rbf, fan_width
        self.pair = nn.Sequential(nn.Linear(2 * d + token_dim, d), nn.Mish(),
                                  nn.Linear(d, d))
        self.gate = nn.Linear(n_rbf, d)
        self.ang = nn.Sequential(nn.Linear(3 * d + fan_width, ang_width), nn.Mish(),
                                 nn.Linear(ang_width, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.Mish(), nn.Linear(d, d))
        _small_init_last(self.fuse)

    def forward(self, h, feats, token, psi, cutoff_edge=6.0, cutoff_trip=4.0):
        """Update atom features using pairwise and triplet interactions

        Parameters:
            h : torch.Tensor
                Atom feature tensor
            feats : dict
                Graph features containing edge and triplet information
            token : torch.Tensor
                Edge feature tokens
            psi : torch.Tensor
                Triplet  features
            cutoff_edge : float
                Maximum distance for edge interactions
            cutoff_trip : float
                Maximum distance for triplet interactions

        Returns:
            torch.Tensor
                Updated atom feature tensor
        """ 
        B, N, _ = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], cutoff_edge) * feats['edge_mask']
        hi, hj = gather_nodes(h, i), gather_nodes(h, j)
        mp = self.pair(torch.cat([hi, hj, token], -1))
        g = torch.sigmoid(self.gate(token[..., 2:2 + self.n_rbf]))
        pair_agg = scatter_mean(mp * g, i, keep, B, N) \
                 + scatter_mean(mp * g, j, keep, B, N)

        it, c, jt = feats['trip_atom'].unbind(-1)
        d1, d2 = feats['trip_d'][..., 1], feats['trip_d'][..., 2]
        lenv = (smooth_cutoff(d1, cutoff_trip)
                * smooth_cutoff(d2, cutoff_trip)) * feats['trip_mask']
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
    def __init__(self, d, token_dim, n_rbf, fan_width=16, heads=4, ang_width=48):
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

    def forward(self, h, feats, token, psi, cutoff_edge=6.0, cutoff_trip=4.0):
        """Update atom features using multiple interaction mechanisms

        Parameters:
            h : torch.Tensor
                Atom feature tensor
            feats : dict
                Graph features containing edge and triplet information
            token : torch.Tensor
                Edge feature tokens
            psi : torch.Tensor
                Triplet interaction features
            cutoff_edge : float
                Maximum distance for edge interactions
            cutoff_trip : float
                Maximum distance for triplet interactions

        Returns:
            torch.Tensor
                Updated atom feature tensor
        """
        B, N, D = h.shape
        i, j = feats['edge_atom'].unbind(-1)
        keep = smooth_cutoff(feats['edge_d'], cutoff_edge) * feats['edge_mask']
        hi, hj = gather_nodes(h, i), gather_nodes(h, j)
        qh = self.q(hi).view(B, -1, self.hd, self.dk)
        kh = self.k(hj).view(B, -1, self.hd, self.dk)
        logit = (qh * kh).sum(-1) / math.sqrt(self.dk) + self.att_bias(token)
        logit = logit.masked_fill(~feats['edge_mask'].unsqueeze(-1), float('-inf'))
        a = logit.exp()
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
        lenv = (smooth_cutoff(d1, cutoff_trip)
                * smooth_cutoff(d2, cutoff_trip)) * feats['trip_mask']
        mt = self.ang(torch.cat([gather_nodes(h, it), gather_nodes(h, jt),
                                 gather_nodes(h, c), psi], -1)) * lenv.unsqueeze(-1)
        ang = scatter_mean(mt, c, feats['trip_mask'].float(), B, N)
        glob = self.glob(h.mean(dim=1, keepdim=True)).expand(B, N, D)
        return self.fuse(torch.cat([attn, conv, ang, glob], -1))


class ForceField(nn.Module):
    """Neural network force field for predicting molecular energies

    Parameters:
        block_cls : nn.Module
            Neural network block used to model atomic interactions
        d : int
            Dimension of the atom feature vectors
        n_rbf : int
            Number of radial basis function features
        cutoff_edge : float
            Maximum distance used for pairwise interactions
        cutoff_trip : float
            Maximum distance used for triplet interactions
        **block_kw : dict
            Additional parameters passed to the selected neural network block

    Returns:
        None
            Initialises the force field model
    """
    def __init__(self, block_cls, d=64, n_rbf=16, cutoff_edge=6.0, cutoff_trip=4.0, **block_kw):
        super().__init__()
        self.cutoff_edge = cutoff_edge
        self.cutoff_trip = cutoff_trip
        self.rbf_edge = RadialBasis(0.4, cutoff_edge, n_rbf)
        self.rbf_trip = RadialBasis(0.4, cutoff_trip, n_rbf)
        token_dim = 2 + n_rbf + 16
        self.edge_tokens = EdgeTokens(self.rbf_edge, self.rbf_trip, fan_width=16,
                                      cutoff_edge=cutoff_edge, cutoff_trip=cutoff_trip)
        self.embed = nn.Sequential(nn.Linear(2, d), nn.Mish(), nn.Linear(d, d))
        self.one_body = nn.Linear(2, 1)
        self.block = block_cls(d, token_dim, n_rbf, **block_kw)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.Mish(), nn.Linear(d, 1))
        nn.init.normal_(self.readout[-1].weight, std=1e-2)
        nn.init.zeros_(self.readout[-1].bias)
        nn.init.zeros_(self.one_body.weight)
        nn.init.zeros_(self.one_body.bias)

    def forward(self, node, pos, cell):
        """Calculate the total energy from atomic features and positions

        Parameters:
            node : torch.Tensor
                Atomic features for each atom
            pos : torch.Tensor
                Atomic positions
            cell : torch.Tensor
                Simulation cell dimensions

        Returns:
            torch.Tensor
                Predicted total energy for each structure
        """
        feats = build_graph_features(pos, cell, self.cutoff_edge, self.cutoff_trip)
        token, psi = self.edge_tokens(feats, pos.shape[1])
        h = self.embed(node)
        dh = self.block(h, feats, token, psi, self.cutoff_edge, self.cutoff_trip)
        h = h + dh
        e_i = self.readout(h).squeeze(-1) + self.one_body(node).squeeze(-1)
        return e_i.sum(-1)


# dictionary of models to be used 
MODELS = {
    'mlp': (MlpBlock, {}),
    'mha': (MhaBlock, {'heads': 4}),
    'conv': (ConvBlock, {}),
    'comb': (CombinedBlock, {'heads': 4}),
}

# Calculator wrapper
class MLFFCalculator(Calculator):
    """ASE Calculator wrapper for the ML force field models.
    Parameters:
        model : nn.Module
            Trained machine learning force field model
        mean_E_shift : float
            Energy shift applied to the predicted energy
        cutoff_edge : float
            Maximum distance used for pairwise interactions
        cutoff_trip : float
            Maximum distance used for triplet interactions
        **kwargs : dict
            Additional arguments passed to the ASE Calculator
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model, mean_E_shift, cutoff_edge=6.0, cutoff_trip=4.0, **kwargs):
        super().__init__(**kwargs)
        self.model = model.to(device).eval()
        self.mean_E_shift = mean_E_shift
        self.cutoff_edge = cutoff_edge
        self.cutoff_trip = cutoff_trip
    
    def calculate(self, atoms=None, properties=['energy', 'forces'], 
                  system_changes=all_changes):
        """Calculate the energy and forces for an atomic structure

        Parameters:
        ------------
            atoms : ase.Atoms
                Atomic structure for which the energy and forces are calculated
            properties : list
                Properties to calculate
            system_changes : list
                Changes made to the atomic structure since the last calculation

        Returns:
        ----------
            None
                Stores the predicted energy and forces in the ASE calculator results
        """
        super().calculate(atoms, properties, system_changes)
        
        elements = atoms.get_chemical_symbols()
        N = len(elements)
        
        node = torch.tensor([LOG_COORDS[el] for el in elements],
                           dtype=torch.float32, device=device).unsqueeze(0)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, 
                          device=device, requires_grad=True).unsqueeze(0)
        cell = torch.tensor(atoms.cell.lengths(), dtype=torch.float32, 
                           device=device).unsqueeze(0)
        
        E_pred = self.model(node, pos, cell)
        forces = -torch.autograd.grad(E_pred.sum(), pos, create_graph=False)[0]
        
        self.results['energy'] = E_pred.item() + self.mean_E_shift
        self.results['forces'] = forces.detach().cpu().numpy().reshape(N, 3)


# ASE Calculator wrapper for GFN2-XtB via tblite's ASE interface.
class GFN2XtBCalculator(Calculator):
    """ASE Calculator wrapper for the GFN2-xTB method

    Parameters:
    ------------
        method : str
            xTB method used for the calculation
        **kwargs : dict
            Additional arguments passed to the ASE Calculator"""
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, method='GFN2-xTB', **kwargs):
        super().__init__(**kwargs)
        self.method = method
    
    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        """Calculate the energy and forces using GFN2-xTB

        Parameters:
        -----------
            atoms : ase.Atoms
                Atomic structure for which the energy and forces are calculated
            properties : list
                Properties to calculate
            system_changes : list
                Changes made to the atomic structure since the last calculation

        Returns:
        ----------
            None
                Stores the calculated energy and forces in the ASE calculator results
        """
        super().calculate(atoms, properties, system_changes)
        
        calc = TBLiteASECalc(method=self.method)
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        
        self.results['energy'] = atoms_copy.get_potential_energy()
        self.results['forces'] = atoms_copy.get_forces()


class ExternalMLFFCalculator(Calculator):
    """Generic wrapper for external MLFF calculators.

    Parameters:
    -------------
        calculator : Calculator
            External ASE calculator used to calculate energies and forces
        **kwargs : dict
            Additional arguments passed to the ASE Calculator
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calculator, **kwargs):
        super().__init__(**kwargs)
        self.external_calc = calculator
    
    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        """Calculate the energy and forces using the external calculator

        Parameters:
        ----------
            atoms : ase.Atoms
                Atomic structure for which the energy and forces are calculated
            properties : list
                Properties to calculate
            system_changes : list
                Changes made to the atomic structure since the last calculation

        Returns:
        ---------
            None
                Stores the calculated energy and forces in the ASE calculator results
        """
        super().calculate(atoms, properties, system_changes)
        
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.external_calc
        
        self.results['energy'] = atoms_copy.get_potential_energy()
        self.results['forces'] = atoms_copy.get_forces()


# load models from checkpoint file.
def load_mlff_model(model_name):
    """Load a trained MLFF model from checkpoint with its name"""
    checkpoint_path = f'best_{model_name}.pt'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    block_cls, block_kw = MODELS[model_name]
    model = ForceField(block_cls, 
                      cutoff_edge=ckpt.get('cutoff_edge', 6.0),
                      cutoff_trip=ckpt.get('cutoff_trip', 4.0),
                      **block_kw).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    mean_E = ckpt.get('mean_E', 0.0)
    
    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"  Parameters: {ckpt.get('n_params', 'N/A')}")
    print(f"  Validation Energy: {ckpt.get('val_E', 'N/A'):.4f} eV")
    print(f"  Validation Forces: {ckpt.get('val_F', 'N/A'):.6f} eV/Å")
    print(f"  Mean energy shift: {mean_E:.2f} eV")
    
    return model, mean_E

# load mace weights from checkpoint
def load_mace_model(model_path="~/Downloads/MACE-*.model", model_name='mace-mp-0', device_str='cuda'):
    
    weights = glob.glob(model_path)
    print(f"Found model at: {model_path}")


def load_all_calculators(include_mace=True, include_ani2x=True, mace_model_path=None):
    """Load all calculators for comparison.
   
    Parameters:
    -----------

        include_mace : bool
            Whether to load the MACE calculator
        include_ani2x : bool
            Whether to load the ANI-2x calculator
        mace_model_path : str
            Path to the MACE model file

    Returns:
    ---------
        dict
            Dictionary containing the loaded calculators
    """
    calculators = {}
    
    # Load custom trained models
    print("\n" + "="*70)
    print("LOADING CUSTOM TRAINED MODELS")
    print("="*70)
    
    for model_name in MODELS.keys():
        try:
            model, mean_E = load_mlff_model(model_name)
            calculators[model_name] = MLFFCalculator(model, mean_E)
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")
    
    # Load public MLFFs
    print("\n" + "="*70)
    print("LOADING PUBLIC MLFFs")
    print("="*70)
    
    if include_mace:
        print("\n--- MACE ---")
        mace_calc = load_mace_model(model_path=mace_model_path)
        if mace_calc is not None:
            calculators['MACE'] = mace_calc
    
    # Load GFN2-XtB reference
    print("\n--- GFN2-XtB Reference ---")
    calculators['GFN2-XtB'] = GFN2XtBCalculator()
    print("Loaded GFN2-XtB calculator")
    
    return calculators


# Static frame accuracy tests
def test_static_accuracy(calculators, test_frames, n_test=None, reference='GFN2-XtB'):
    """Compare predicted energies and forces with xtb reference
    
    Parameters:
    ---------
    calculators: Dictionary
        dictionary containing calculators of each model 
    test_frames
        test structures used for comparison 
    n_test : int
        maximum number of test frames to use 
    reference: str
        name of reference xtb calculator 

    Returns:
    ---------
    stats_df : Dataframe
        Dataframe containing errors for each model 
    """
    print("\n" + "="*70)
    print("STATIC ACCURACY TESTS")
    print("="*70)
    
    if n_test is not None:
        test_frames = test_frames[:n_test]
    
    print(f"Computing {reference} ground truth...")
    ref_calc = calculators[reference]
    
    ref_data = []
    for frame_idx, atoms in enumerate(test_frames):
        print(f"  Computing reference for frame {frame_idx + 1}/{len(test_frames)}...")
        atoms_copy = atoms.copy()
        atoms_copy.calc = ref_calc
        E_ref = atoms_copy.get_potential_energy()
        F_ref = atoms_copy.get_forces()
        ref_data.append((E_ref, F_ref))
    
    results = []
    model_names = [name for name in calculators.keys() if name != reference]
    
    for model_name in model_names:
        print(f"Computing {model_name} predictions...")
        calc = calculators[model_name]
        
        for frame_idx, (atoms, (E_ref, F_ref)) in enumerate(zip(test_frames, ref_data)):
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            E_pred = atoms_copy.get_potential_energy()
            F_pred = atoms_copy.get_forces()
            
            results.append({
                'frame': frame_idx,
                'model': model_name,
                'E_ref': E_ref,
                'E_pred': E_pred,
                'dE': E_pred - E_ref,
                'F_ref': F_ref.flatten(),
                'F_pred': F_pred.flatten(),
                'dF': (F_pred - F_ref).flatten()
            })
    
    df = pd.DataFrame(results)
    
    stats = []
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        
        dE = model_df['dE'].values
        dF = np.concatenate(model_df['dF'].values)
        F_ref = np.concatenate(model_df['F_ref'].values)
        
        e_mae = np.mean(np.abs(dE))
        e_rmse = np.sqrt(np.mean(dE**2))
        e_max = np.max(np.abs(dE))
        
        f_mae = np.mean(np.abs(dF))
        f_rmse = np.sqrt(np.mean(dF**2))
        f_max = np.max(np.abs(dF))
        
        f_ss_res = np.sum(dF**2)
        f_ss_tot = np.sum((F_ref - np.mean(F_ref))**2)
        f_r2 = 1 - f_ss_res / f_ss_tot if f_ss_tot > 0 else 0.0
        
        stats.append({
            'Model': model_name,
            'Energy MAE (eV)': e_mae,
            'Energy RMSE (eV)': e_rmse,
            'Energy Max Error (eV)': e_max,
            'Force MAE (eV/Å)': f_mae,
            'Force RMSE (eV/Å)': f_rmse,
            'Force Max Error (eV/Å)': f_max,
            'Force R²': f_r2
        })
        
        print(f"\n{model_name.upper()}:")
        print(f"  Energy MAE:  {e_mae:.4f} eV")
        print(f"  Energy RMSE: {e_rmse:.4f} eV")
        print(f"  Energy Max:  {e_max:.4f} eV")
        print(f"  Force MAE:   {f_mae:.4f} eV/Å")
        print(f"  Force RMSE:  {f_rmse:.4f} eV/Å")
        print(f"  Force Max:   {f_max:.4f} eV/Å")
        print(f"  Force R²:    {f_r2:.4f}")
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('test_results/csvs/static_accuracy.csv', index=False)
    df.to_csv('test_results/csvs/static_predictions.csv', index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        axes[0].hist(model_df['dE'], bins=30, alpha=0.6, label=model_name)
    
    axes[0].set_xlabel('Energy Error (eV)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Energy Error Distribution')
    axes[0].legend()
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        dF = np.concatenate(model_df['dF'].values)
        axes[1].hist(dF, bins=50, alpha=0.6, label=model_name, density=True)
    
    axes[1].set_xlabel('Force Error (eV/Å)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Force Error Distribution')
    axes[1].legend()
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('test_results/plots/static_accuracy.png', dpi=150)
    plt.close()
    
    return stats_df


# Energy conservation test in NVE ensemble
def test_energy_conservation(calculators, start_atoms, steps=2000, timestep_fs=0.5,
                             skip_models=None):
    """tests whether total energy is conserved during NVE MD

    Parameters:
    ----------

        calculators : dictionary 
            dictionary containing the calculators for each model
        start_atoms : ASE.atoms 
            Initial atomic structure for the simulation
        steps: int
            Number of MD steps to perform
        timestep_fs : int
            MD timestep in femtoseconds
        skip_models : str
            List of models to exclude from the test

    Returns:
    ----------

        results: Dictionary 
            Dictionary containing the energy values for each model
    """
    print("\n" + "="*70)
    print("ENERGY CONSERVATION TEST (NVE MD)")
    print("="*70)
    
    if skip_models is None:
        skip_models = ['GFN2-XtB']
    
    results = {'time_ps': np.arange(steps) * timestep_fs / 1000}
    
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    for model_name in model_names:
        print(f"Running {model_name}...")
        
        at = start_atoms.copy()
        MaxwellBoltzmannDistribution(at, temperature_K=300)
        at.calc = calculators[model_name]
        
        dyn = VelocityVerlet(at, timestep=timestep_fs * 0.001)
        
        epot, ekin = [], []
        
        for step in range(steps):
            dyn.run(1)
            epot.append(at.get_potential_energy())
            ekin.append(at.get_kinetic_energy())
        
        epot = np.array(epot)
        ekin = np.array(ekin)
        etot = epot + ekin
        
        results[f'{model_name}_epot'] = epot
        results[f'{model_name}_ekin'] = ekin
        results[f'{model_name}_etot'] = etot
        
        energy_range = etot.max() - etot.min()
        energy_std = etot.std()
        
        print(f"  {model_name}:")
        print(f"    Energy range: {energy_range:.6e} eV")
        print(f"    Energy std:   {energy_std:.6e} eV")
    
    pd.DataFrame(results).to_csv('test_results/csvs/energy_conservation.csv', index=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for model_name in model_names:
        axes[0].plot(results['time_ps'], results[f'{model_name}_etot'], 
                    label=model_name, alpha=0.8)
    
    axes[0].set_xlabel('Time (ps)')
    axes[0].set_ylabel('Total Energy (eV)')
    axes[0].set_title('Total Energy Conservation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for model_name in model_names:
        axes[1].plot(results['time_ps'], 
                    results[f'{model_name}_etot'] - results[f'{model_name}_etot'][0],
                    label=model_name, alpha=0.8)
    
    axes[1].set_xlabel('Time (ps)')
    axes[1].set_ylabel('ΔE (eV)')
    axes[1].set_title('Energy Drift')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results/plots/energy_conservation.png', dpi=150)
    plt.close()
    
    return results


# Test that sum of forces equals zero
def test_force_conservation(calculators, start_atoms, steps=2000, timestep_fs=0.5,
                           skip_models=None):
    """Test whether the net force on the system remains close to zero

    Parameters:
        calculators: Dictionary
            Dictionary containing the calculators for each model
        start_atoms: ASE.atoms
             Initial atomic structure for the simulation
        steps: int
            Number of MD steps to perform
        timestep_fs: int/float
            MD timestep in femtoseconds
        skip_models: str
            List of models to exclude from the test

    Returns:
        results: Dictionary containing the net force values for each model

    """
    print("\n" + "="*70)
    print("FORCE CONSERVATION TEST (ΣF = 0)")
    print("="*70)
    
    if skip_models is None:
        skip_models = ['GFN2-XtB']
    
    results = {'time_ps': np.arange(steps) * timestep_fs / 1000}
    
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    for model_name in model_names:
        print(f"Running {model_name}...")
        
        at = start_atoms.copy()
        MaxwellBoltzmannDistribution(at, temperature_K=300)
        at.calc = calculators[model_name]
        
        dyn = VelocityVerlet(at, timestep=timestep_fs * 0.001)
        
        force_sums = []
        
        for step in range(steps):
            dyn.run(1)
            forces = at.get_forces()
            total_force = np.sum(forces, axis=0)
            force_sums.append(np.linalg.norm(total_force))
        
        force_sums = np.array(force_sums)
        results[f'{model_name}_force_sum'] = force_sums
        
        print(f"  {model_name}:")
        print(f"    Max |ΣF|: {force_sums.max():.6e} eV/Å")
        print(f"    Mean |ΣF|: {force_sums.mean():.6e} eV/Å")
        
        passed = force_sums.max() < 1e-5
        print(f"    Status: {'PASS' if passed else 'FAIL'}")
    
    pd.DataFrame(results).to_csv('test_results/csvs/force_conservation.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for model_name in model_names:
        ax.semilogy(results['time_ps'], results[f'{model_name}_force_sum'],
                   label=model_name, alpha=0.8)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('|ΣF| (eV/Å)')
    ax.set_title('Force Conservation (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(1e-5, color='r', linestyle='--', label='Tolerance')
    
    plt.tight_layout()
    plt.savefig('test_results/plots/force_conservation.png', dpi=150)
    plt.close()
    
    return results


# Invariance test energy and forces are invariant under translation.
def test_translational_invariance(calculators, test_atoms, n_tests=10, max_shift=10.0,
                                  skip_models=None):
    """Test whether energies and forces are unchanged after translation

    Parameters:
        calculators: Dictionary 
            dictionary containing the calculators for each model
        test_atoms: ASE.atoms
            Atomic structure used for the test
        n_tests: int
            Number of random translations to perform
        max_shift: int/float
            Maximum translation distance in angstroms
        skip_models: str
            List of models to exclude from the test

    Returns:
        df: pd.Dataframe
            DataFrame containing energy and force deviations
    """
    
    print("\n" + "="*70)
    print("TRANSLATIONAL INVARIANCE TEST")
    print("="*70)
    
    if skip_models is None:
        skip_models = ['GFN2-XtB']
    
    results = []
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    for model_name in model_names:
        print(f"Testing {model_name}...")
        
        calc = calculators[model_name]
        
        at_orig = test_atoms.copy()
        at_orig.calc = calc
        E_orig, F_orig = at_orig.get_potential_energy(), at_orig.get_forces()
        
        energy_devs, force_devs = [], []
        
        for test_idx in range(n_tests):
            shift = np.random.uniform(-max_shift, max_shift, size=3)
            
            at_trans = test_atoms.copy()
            at_trans.translate(shift)
            at_trans.calc = calc
            E_trans, F_trans = at_trans.get_potential_energy(), at_trans.get_forces()
            
            energy_dev = abs(E_trans - E_orig)
            force_dev = np.max(np.abs(F_trans - F_orig))
            
            energy_devs.append(energy_dev)
            force_devs.append(force_dev)
            
            results.append({
                'model': model_name,
                'test': test_idx,
                'energy_dev': energy_dev,
                'force_dev': force_dev
            })
        
        energy_devs = np.array(energy_devs)
        force_devs = np.array(force_devs)
        
        print(f"  {model_name}:")
        print(f"    Max energy deviation: {energy_devs.max():.6e} eV")
        print(f"    Max force deviation:  {force_devs.max():.6e} eV/Å")
        
        e_passed = energy_devs.max() < 1e-4
        f_passed = force_devs.max() < 1e-4
        print(f"    Energy: {'PASS' if e_passed else 'FAIL'}")
        print(f"    Forces: {'PASS' if f_passed else 'FAIL'}")
    
    df = pd.DataFrame(results)
    df.to_csv('test_results/csvs/translational_invariance.csv', index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        axes[0].hist(model_df['energy_dev'], bins=10, alpha=0.6, label=model_name)
        axes[1].hist(model_df['force_dev'], bins=10, alpha=0.6, label=model_name)
    
    axes[0].set_xlabel('Energy Deviation (eV)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Translational Invariance: Energy')
    axes[0].legend()
    
    axes[1].set_xlabel('Force Deviation (eV/Å)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Translational Invariance: Forces')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('test_results/plots/translational_invariance.png', dpi=150)
    plt.close()
    
    return df


# Rotational invariance test 
def test_rotational_invariance(calculators, test_atoms, n_tests=10, skip_models=None):
    """Test whether energies and forces are unchanged after rotation

        Parameters:
            calculators: Dictionary 
                dictionary containing the calculators for each model
            test_atoms: ASE.atoms
                Atomic structure used for the test
            n_tests: int
                Number of random translations to perform
            max_shift: int/float
                Maximum translation distance in angstroms
            skip_models: str
                List of models to exclude from the test

        Returns:
            df: pd.Dataframe
                DataFrame containing energy and force deviations
        """

    print("\n" + "="*70)
    print("ROTATIONAL INVARIANCE TEST")
    print("="*70)
    
    if skip_models is None:
        skip_models = ['GFN2-XtB']
    
    results = []
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    for model_name in model_names:
        print(f"Testing {model_name}...")
        
        calc = calculators[model_name]
        
        at_orig = test_atoms.copy()
        at_orig.calc = calc
        E_orig, F_orig = at_orig.get_potential_energy(), at_orig.get_forces()
        
        energy_devs, force_mag_devs, force_dir_sims = [], [], []
        
        for test_idx in range(n_tests):
            rot = Rotation.random()
            R = rot.as_matrix()
            
            at_rot = test_atoms.copy()
            com = at_rot.get_center_of_mass()
            positions = at_rot.get_positions()
            rotated_positions = (positions - com) @ R.T + com
            at_rot.set_positions(rotated_positions)
            at_rot.calc = calc
            
            E_rot, F_rot = at_rot.get_potential_energy(), at_rot.get_forces()
            
            energy_dev = abs(E_rot - E_orig)
            
            F_orig_mag = np.linalg.norm(F_orig, axis=1)
            F_rot_mag = np.linalg.norm(F_rot, axis=1)
            force_mag_dev = np.max(np.abs(F_rot_mag - F_orig_mag))
            
            F_expected = F_orig @ R.T
            dot_products = np.sum(F_rot * F_expected, axis=1)
            F_rot_mag_safe = np.maximum(F_rot_mag, 1e-12)
            F_exp_mag = np.linalg.norm(F_expected, axis=1)
            F_exp_mag_safe = np.maximum(F_exp_mag, 1e-12)
            cos_sim = dot_products / (F_rot_mag_safe * F_exp_mag_safe)
            min_cos_sim = cos_sim.min()
            
            energy_devs.append(energy_dev)
            force_mag_devs.append(force_mag_dev)
            force_dir_sims.append(min_cos_sim)
            
            results.append({
                'model': model_name,
                'test': test_idx,
                'energy_dev': energy_dev,
                'force_mag_dev': force_mag_dev,
                'force_dir_cos_sim': min_cos_sim
            })
        
        energy_devs = np.array(energy_devs)
        force_mag_devs = np.array(force_mag_devs)
        force_dir_sims = np.array(force_dir_sims)
        
        print(f"  {model_name}:")
        print(f"    Max energy deviation:      {energy_devs.max():.6e} eV")
        print(f"    Max force magnitude dev:   {force_mag_devs.max():.6e} eV/Å")
        print(f"    Min force direction cos:   {force_dir_sims.min():.6f}")
        
        e_passed = energy_devs.max() < 1e-4
        f_mag_passed = force_mag_devs.max() < 1e-4
        f_dir_passed = force_dir_sims.min() > 0.99
        
        print(f"    Energy invariance:  {'PASS' if e_passed else 'FAIL'}")
        print(f"    Force mag equiv:    {'PASS' if f_mag_passed else 'FAIL'}")
        print(f"    Force dir equiv:    {'PASS' if f_dir_passed else 'FAIL'}")
    
    df = pd.DataFrame(results)
    df.to_csv('test_results/csvs/rotational_invariance.csv', index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        axes[0].hist(model_df['energy_dev'], bins=10, alpha=0.6, label=model_name)
        axes[1].hist(model_df['force_mag_dev'], bins=10, alpha=0.6, label=model_name)
        axes[2].hist(model_df['force_dir_cos_sim'], bins=10, alpha=0.6, label=model_name)
    
    axes[0].set_xlabel('Energy Deviation (eV)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Energy Invariance')
    axes[0].legend()
    
    axes[1].set_xlabel('Force Magnitude Deviation (eV/Å)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Force Magnitude Equivariance')
    axes[1].legend()
    
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Force Direction Equivariance')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('test_results/plots/rotational_invariance.png', dpi=150)
    plt.close()
    
    return df


# Calculate radial distribution function for O and H
def calculate_rdf(atoms, idx, rmax, nbins):
    """Calculate the radial distribution function for atoms 

    Parameters:
        atoms: ASE.atoms
            Atomic structure used to calculate the RDF
        idx: int
            Indices of the atoms included in the RDF calculation
        rmax: int
            Maximum distance considered in angstroms
        nbins: int
            Number of distance bins

    Returns:
        r: np.array
            Array containing the radial distances
        rdf: np.array
            Array containing the radial distribution function values
    """
    pos = atoms.positions[idx]
    N = len(pos)
    V = atoms.get_volume()
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    
    cell = atoms.get_cell().array
    inv_cell = np.linalg.inv(cell)
    diff_frac = np.einsum('ijk,kl->ijl', diff, inv_cell)
    diff_frac -= np.round(diff_frac)
    diff = np.einsum('ijk,kl->ijl', diff_frac, cell)
    
    dists = np.linalg.norm(diff, axis=-1)
    dists = dists[np.triu_indices(N, k=1)]
    dists = dists[dists <= rmax]
    
    bins = np.linspace(0, rmax, nbins + 1)
    hist, _ = np.histogram(dists, bins=bins)
    
    r = 0.5 * (bins[1:] + bins[:-1])
    shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    norm = (N * (N - 1) / 2.0) / V * shell_volumes
    norm[norm == 0] = 1.0
    
    return r, hist / norm


def test_rdf_comparison(calculators, start_atoms, steps=2000, rmax=8.0, nbins=50,
                       skip_models=None):
    """Compare radial distribution functions generated by different models
    
        Parameters:
            calculators : Dictionary 
                Dictionary containing the calculators for each model
            start_atoms : ASE.atoms
                Initial atomic structure for the simulation
            steps : int
                Number of MD steps to perform
            rmax: int,float
                Maximum RDF distance in angstroms
            nbins: int
                Number of RDF distance bins
            skip_models: str
                List of models to exclude from the test
    
        Returns:
            results: Dictionary 
            Dictionary containing the RDF for each model
        """
    
    print("\n" + "="*70)
    print("RDF COMPARISON TEST")
    print("="*70)
    
    if skip_models is None:
        skip_models = []
    
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    n_molecules = len(start_atoms) // 3
    o_indices = [i * 3 for i in range(n_molecules)]
    
    bins = np.linspace(0, rmax, nbins + 1)
    r = 0.5 * (bins[1:] + bins[:-1])
    
    results = {'r_Angstrom': r}
    
    for model_name in model_names:
        print(f"Computing {model_name} RDF...")
        
        at = start_atoms.copy()
        MaxwellBoltzmannDistribution(at, temperature_K=300)
        at.calc = calculators[model_name]
        
        dyn = VelocityVerlet(at, timestep=0.5 * 0.001)
        traj = []
        
        for step in range(steps):
            dyn.run(10)
            if step >= 50:
                traj.append(at.copy())
        
        rdf = np.zeros(nbins)
        for frame in traj:
            r_frame, rdf_frame = calculate_rdf(frame, o_indices, rmax, nbins)
            rdf += rdf_frame
        rdf /= len(traj)
        
        results[f'{model_name}_RDF'] = rdf
        
        if 'GFN2-XtB_RDF' in results:
            rdf_diff = np.abs(rdf - results['GFN2-XtB_RDF'])
            print(f"  {model_name}: Mean |ΔRDF| = {np.mean(rdf_diff):.4f}")
        else:
            print(f"  {model_name}: Mean g(r) peak = {rdf.max():.4f}")
    
    pd.DataFrame(results).to_csv('test_results/csvs/rdf_comparison.csv', index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        linestyle = '-' if model_name == 'GFN2-XtB' else '--'
        linewidth = 2.5 if model_name == 'GFN2-XtB' else 1.5
        axes[0].plot(r, results[f'{model_name}_RDF'], linestyle, 
                    label=model_name, color=colors[idx], linewidth=linewidth)
    
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title('O-O Radial Distribution Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'GFN2-XtB_RDF' in results:
        for idx, model_name in enumerate(model_names):
            if model_name != 'GFN2-XtB':
                axes[1].plot(r, np.abs(results[f'{model_name}_RDF'] - results['GFN2-XtB_RDF']), 
                            label=model_name, color=colors[idx])
    
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('|Δg(r)|')
    axes[1].set_title('RDF Deviation from GFN2-XtB')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results/plots/rdf_comparison.png', dpi=150)
    plt.close()
    
    return results


# check count of hydrogen bonds.
def count_hbonds(atoms, o_indices, h_indices, r_oh=2.5, r_oo=3.5, angle_min=120):
    """Count hydrogen bonds using distance and angle criteria

    Parameters:
        atoms: ASE.atoms
            Atomic structure used to identify hydrogen bonds
        o_indices: int
            Indices of oxygen atoms
        h_indices: int
            Indices of hydrogen atoms
        r_oh: float
            Maximum hydrogen-oxygen distance in angstroms
        r_oo: float
            Maximum oxygen-oxygen distance in angstroms
        angle_min: float
            Minimum hydrogen bond angle in degrees

    Returns:
        count: int,float
            Number of hydrogen bonds identified
    """
    count = 0
    for o_idx in o_indices:
        for h_idx in h_indices:
            if abs(h_idx - o_idx) <= 2:
                continue
            
            r_oh_val = atoms.get_distance(o_idx, h_idx, mic=True)
            if r_oh_val > r_oh:
                continue
            
            o_donate_idx = (h_idx // 3) * 3
            r_oo_val = atoms.get_distance(o_idx, o_donate_idx, mic=True)
            
            if r_oo_val > r_oo:
                continue
            
            angle = atoms.get_angle(h_idx, o_idx, o_donate_idx, mic=True)
            if angle > angle_min:
                count += 1
    
    return count


def test_hydrogen_bonds(calculators, start_atoms, steps=2000, skip_models=None):
    """Compare hydrogen bond statistics generated by different models

    Parameters:
        calculators: Dictionary 
            Dictionary containing the calculators for each model
        start_atoms: ASE.atoms
            Initial atomic structure for the simulation
        steps: int
            Number of MD steps to perform
        skip_models: str
            List of models to exclude from the test

    Returns:
        results: Dictionary
            Dictionary containing hydrogen bond statistics for each model
    """
    print("\n" + "="*70)
    print("HYDROGEN BOND ANALYSIS")
    print("="*70)
    
    if skip_models is None:
        skip_models = []
    
    model_names = [name for name in calculators.keys() if name not in skip_models]
    
    n_molecules = len(start_atoms) // 3
    o_indices = [i * 3 for i in range(n_molecules)]
    h_indices = [i for i in range(len(start_atoms)) if i % 3 != 0]
    
    time_ps = np.arange(steps) * 5.0
    results = {'time_ps': time_ps}
    
    for model_name in model_names:
        print(f"Computing {model_name} H-bonds...")
        
        at = start_atoms.copy()
        MaxwellBoltzmannDistribution(at, temperature_K=300)
        at.calc = calculators[model_name]
        
        dyn = VelocityVerlet(at, timestep=0.5 * 0.001)
        hbonds = []
        
        for step in range(steps):
            dyn.run(10)
            n_hbonds = count_hbonds(at, o_indices, h_indices)
            hbonds.append(n_hbonds / n_molecules)
        
        results[f'{model_name}_HBonds'] = hbonds
        
        hbonds_eq = np.array(hbonds[50:])
        print(f"  {model_name}: Mean H-bonds = {np.mean(hbonds_eq):.3f} ± {np.std(hbonds_eq):.3f}")
    
    pd.DataFrame(results).to_csv('test_results/csvs/hydrogen_bonds.csv', index=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        linestyle = '-' if model_name == 'GFN2-XtB' else '--'
        linewidth = 2.5 if model_name == 'GFN2-XtB' else 1.5
        axes[0].plot(time_ps, results[f'{model_name}_HBonds'], linestyle,
                    label=model_name, color=colors[idx], linewidth=linewidth)
    
    axes[0].set_xlabel('Time (ps)')
    axes[0].set_ylabel('H-Bonds / Molecule')
    axes[0].set_title('Hydrogen Bond Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for idx, model_name in enumerate(model_names):
        axes[1].hist(results[f'{model_name}_HBonds'][50:], bins=30, alpha=0.5, 
                    density=True, label=model_name, color=colors[idx])
    
    axes[1].set_xlabel('H-Bonds / Molecule')
    axes[1].set_ylabel('Density')
    axes[1].set_title('H-Bond Distribution (Equilibrium)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('test_results/plots/hydrogen_bonds.png', dpi=150)
    plt.close()
    
    return results


def main():
    print("="*70)
    print("MLFF vs GFN2-XtB PHYSICS TESTING FRAMEWORK")
    print("="*70)
    
    print("\nLoading test data...")
    test_file = 'water_dataset_64.extxyz'
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    all_frames = read(test_file, index=':')
    print(f"Loaded {len(all_frames)} frames")
    
    np.random.seed(42)
    test_indices = np.random.choice(len(all_frames), min(50, len(all_frames)), replace=False)
    test_frames = [all_frames[i] for i in test_indices]
    
    # Load all calculators
    calculators = load_all_calculators(
        include_mace=True,
        include_ani2x=True,
        mace_model_path=None  # Will auto-detect from ~/Downloads/
    )
    
    print(f"\nCalculators loaded: {list(calculators.keys())}")
    
    print("\n" + "#"*70)
    print("# STARTING TEST SUITE")
    print("#"*70)
    
    # Models to skip in MD tests
    md_skip = ['GFN2-XtB']
    
    # Static accuracy
    static_stats = test_static_accuracy(calculators, test_frames, n_test=20)
    
    # Energy conservation
    energy_results = test_energy_conservation(calculators, test_frames[0].copy(), 
                                              steps=2000, skip_models=md_skip)
    
    # Force conservation
    force_results = test_force_conservation(calculators, test_frames[0].copy(), 
                                            steps=2000, skip_models=md_skip)
    
    # Translational invariance
    trans_results = test_translational_invariance(calculators, test_frames[0].copy(), 
                                                  n_tests=10, skip_models=md_skip)
    
    # Rotational invariance
    rot_results = test_rotational_invariance(calculators, test_frames[0].copy(), 
                                             n_tests=10, skip_models=md_skip)
    
    # RDF comparison
    rdf_results = test_rdf_comparison(calculators, test_frames[0].copy(), 
                                      steps=2000, skip_models=[])
    
    # Hydrogen bond analysis
    hbond_results = test_hydrogen_bonds(calculators, test_frames[0].copy(), 
                                        steps=2000, skip_models=[])
    
    # Summary report
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    ml_model_names = [name for name in calculators.keys() if name not in ['GFN2-XtB']]
    
    summary = []
    
    for model_name in ml_model_names:
        # Get static accuracy stats
        if model_name in static_stats['Model'].values:
            e_mae = static_stats.loc[static_stats['Model'] == model_name, 'Energy MAE (eV)'].values[0]
            f_mae = static_stats.loc[static_stats['Model'] == model_name, 'Force MAE (eV/Å)'].values[0]
            f_r2 = static_stats.loc[static_stats['Model'] == model_name, 'Force R²'].values[0]
        else:
            e_mae = f_mae = f_r2 = np.nan
        
        # Get energy conservation
        if f'{model_name}_etot' in energy_results:
            e_drift = energy_results[f'{model_name}_etot'].max() - energy_results[f'{model_name}_etot'].min()
        else:
            e_drift = np.nan
        
        # Get force conservation
        if f'{model_name}_force_sum' in force_results:
            max_fsum = force_results[f'{model_name}_force_sum'].max()
        else:
            max_fsum = np.nan
        
        # Get invariance
        if model_name in trans_results['model'].values:
            max_trans_e = trans_results.loc[trans_results['model'] == model_name, 'energy_dev'].max()
        else:
            max_trans_e = np.nan
        
        if model_name in rot_results['model'].values:
            max_rot_e = rot_results.loc[rot_results['model'] == model_name, 'energy_dev'].max()
        else:
            max_rot_e = np.nan
        
        # Get RDF deviation
        if f'{model_name}_RDF' in rdf_results and 'GFN2-XtB_RDF' in rdf_results:
            mean_rdf_dev = np.mean(np.abs(rdf_results[f'{model_name}_RDF'] - rdf_results['GFN2-XtB_RDF']))
        else:
            mean_rdf_dev = np.nan
        
        # Get H-bonds
        if f'{model_name}_HBonds' in hbond_results:
            mean_hbonds = np.mean(hbond_results[f'{model_name}_HBonds'][50:])
        else:
            mean_hbonds = np.nan
        
        summary.append({
            'Model': model_name,
            'Energy MAE (eV)': e_mae,
            'Force MAE (eV/Å)': f_mae,
            'Force R²': f_r2,
            'Energy Drift (eV)': e_drift,
            'Max |ΣF| (eV/Å)': max_fsum,
            'Max Trans E Dev (eV)': max_trans_e,
            'Max Rot E Dev (eV)': max_rot_e,
            'Mean |ΔRDF|': mean_rdf_dev,
            'Mean H-Bonds': mean_hbonds
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('test_results/csvs/test_summary.csv', index=False)
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    model_names = summary_df['Model'].values
    x = np.arange(len(model_names))
    width = 0.6
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    axes[0, 0].bar(x, summary_df['Energy MAE (eV)'], width, color=colors)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('eV')
    axes[0, 0].set_title('Energy MAE')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(x, summary_df['Force MAE (eV/Å)'], width, color=colors)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('eV/Å')
    axes[0, 1].set_title('Force MAE')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[0, 2].bar(x, summary_df['Force R²'], width, color=colors)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].set_title('Force R²')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Log scale for drift/force sum
    e_drift = summary_df['Energy Drift (eV)'].values
    valid = ~np.isnan(e_drift)
    if valid.any():
        axes[1, 0].bar(x[valid], e_drift[valid], width, color=colors[valid])
        axes[1, 0].set_yscale('log')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('eV')
    axes[1, 0].set_title('Energy Drift (NVE MD)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    max_fsum = summary_df['Max |ΣF| (eV/Å)'].values
    valid = ~np.isnan(max_fsum)
    if valid.any():
        axes[1, 1].bar(x[valid], max_fsum[valid], width, color=colors[valid])
        axes[1, 1].set_yscale('log')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('eV/Å')
    axes[1, 1].set_title('Max |ΣF| (Force Conservation)')
    axes[1, 1].axhline(1e-5, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    mean_rdf = summary_df['Mean |ΔRDF|'].values
    valid = ~np.isnan(mean_rdf)
    if valid.any():
        axes[1, 2].bar(x[valid], mean_rdf[valid], width, color=colors[valid])
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 2].set_ylabel('|Δg(r)|')
    axes[1, 2].set_title('Mean RDF Deviation')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_results/plots/test_summary.png', dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: test_results/")
    print("  - plots/   : Visualization of all tests")
    print("  - csvs/    : Numerical results in CSV format")


if __name__ == "__main__":
    main()
