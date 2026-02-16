"""Low-level mathematical functions for time-series feature computation."""

from collections import Counter
from itertools import permutations
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

EPSILON_ZERO = 1e-10
EPSILON_LOG = 1e-12


def build_embedded_vectors(
    signal: NDArray[np.float64], 
    embedding_dim: int, 
    time_delay: int
) -> List[List[float]]:
    """
    Create time-delay embedded vectors from signal.
    
    Args:
        signal: Input time series
        embedding_dim: Embedding dimension (m)
        time_delay: Time delay between samples (tau)
        
    Returns:
        List of embedded vectors
        
    Raises:
        ValueError: If signal is too short for given parameters
    """
    n_vectors = len(signal) - (embedding_dim - 1) * time_delay
    if n_vectors <= 0:
        raise ValueError(
            f"Signal length {len(signal)} too short for m={embedding_dim}, tau={time_delay}"
        )
    
    return [
        [signal[i + j * time_delay] for j in range(embedding_dim)]
        for i in range(n_vectors)
    ]


def compute_ordinal_patterns(
    embedded_vectors: List[List[float]]
) -> List[Tuple[int, ...]]:
    """
    Convert embedded vectors to ordinal patterns.
    
    Args:
        embedded_vectors: List of embedded vectors
        
    Returns:
        List of ordinal patterns as tuples
    """
    return [tuple(np.argsort(vector)) for vector in embedded_vectors]


def build_probability_distribution(
    ordinal_patterns: List[Tuple[int, ...]],
    embedding_dim: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """
    Build probability distributions from ordinal patterns.
    
    Args:
        ordinal_patterns: List of ordinal patterns
        embedding_dim: Embedding dimension
        
    Returns:
        Tuple of (observed_prob, full_prob, n_permutations)
    """
    pattern_counts = Counter(ordinal_patterns)
    total_patterns = len(ordinal_patterns)
    
    observed_prob = np.array([
        pattern_counts[pattern] / (total_patterns + EPSILON_ZERO)
        for pattern in pattern_counts.keys()
    ])
    
    all_patterns = list(permutations(range(embedding_dim)))
    n_permutations = len(all_patterns)
    
    full_prob = np.zeros(n_permutations)
    for idx, pattern in enumerate(all_patterns):
        if pattern in pattern_counts:
            full_prob[idx] = pattern_counts[pattern] / (total_patterns + EPSILON_ZERO)
    
    return observed_prob, full_prob, n_permutations


def compute_shannon_entropy(distribution: NDArray) -> float:
    """
    Calculate Shannon entropy of probability distribution.
    
    Args:
        distribution: Probability distribution
        
    Returns:
        Shannon entropy value
    """
    return float(-np.sum(distribution * np.log(distribution + EPSILON_LOG)))


def compute_js_divergence(
    prob_full: NDArray[np.float64],
    prob_uniform: NDArray[np.float64],
    entropy_p: float,
    entropy_uniform: float
) -> float:
    """
    Calculate Jensen-Shannon divergence between distributions.
    
    Args:
        prob_full: Full probability distribution
        prob_uniform: Uniform probability distribution
        entropy_p: Entropy of prob_full
        entropy_uniform: Entropy of prob_uniform
        
    Returns:
        Jensen-Shannon divergence value
    """
    mixture = ((prob_full + prob_uniform) / 2).astype(np.float64)
    entropy_mixture = compute_shannon_entropy(mixture)
    return float(entropy_mixture - 0.5 * entropy_p - 0.5 * entropy_uniform)


def compute_max_divergence(
    prob_uniform: NDArray[np.float64],
    entropy_uniform: float
) -> float:
    """
    Calculate maximum possible Jensen-Shannon divergence.
    
    Args:
        prob_uniform: Uniform probability distribution
        entropy_uniform: Entropy of uniform distribution
        
    Returns:
        Maximum divergence value
    """
    prob_delta = np.zeros(len(prob_uniform))
    prob_delta[0] = 1.0
    
    mixture_max = ((prob_delta + prob_uniform) / 2).astype(np.float64)
    entropy_mixture_max = compute_shannon_entropy(mixture_max)
    
    return float(entropy_mixture_max - 0.5 * entropy_uniform)