"""
Low-level mathematical helper functions for feature computations.
These are internal functions used by features.py.
"""
import numpy as np
from collections import Counter
from itertools import permutations


def _construct_embedded_vectors(y, m, tau):
    """Construct embedded vectors from time series using time delay embedding."""
    n_vectors = len(y) - (m - 1) * tau
    if n_vectors <= 0:
        raise ValueError("Time series too short for given m and tau")
    
    return [[y[i + j * tau] for j in range(m)] for i in range(n_vectors)]


def _compute_ordinal_patterns(embedded_vectors):
    """Convert embedded vectors to ordinal patterns (permutations)."""
    return [tuple(np.argsort(vector)) for vector in embedded_vectors]


def _build_probability_distributions(ordinal_patterns, m):
    """Build probability distributions from ordinal patterns."""
    pattern_counts = Counter(ordinal_patterns)
    total_patterns = len(ordinal_patterns)
    eps = 1e-10
    
    P = np.array([pattern_counts[pattern] / (total_patterns + eps) 
                  for pattern in pattern_counts.keys()])
    
    all_patterns = list(permutations(range(m)))
    n_permutations = len(all_patterns)
    
    P_full = np.zeros(n_permutations)
    for idx, pattern in enumerate(all_patterns):
        if pattern in pattern_counts:
            P_full[idx] = pattern_counts[pattern] / (total_patterns + eps)
    
    return P, P_full, n_permutations


def _compute_shannon_entropy(distribution):
    """Compute Shannon entropy of a probability distribution."""
    return -np.sum(distribution * np.log(distribution + 1e-12))


def _compute_jensen_shannon_divergence(P_full, P_uniform, S_P, S_Pu):
    """Compute Jensen-Shannon divergence between two distributions."""
    M = (P_full + P_uniform) / 2
    S_M = _compute_shannon_entropy(M)
    return S_M - 0.5 * S_P - 0.5 * S_Pu


def _compute_max_divergence(P_uniform, S_Pu):
    """Compute maximum possible Jensen-Shannon divergence."""
    P_delta = np.zeros(len(P_uniform))
    P_delta[0] = 1.0
    M_max = (P_delta + P_uniform) / 2
    S_M_max = _compute_shannon_entropy(M_max)
    S_delta = 0.0
    return S_M_max - 0.5 * S_delta - 0.5 * S_Pu