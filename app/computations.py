from collections import Counter
from itertools import permutations

import numpy as np
from numpy.typing import NDArray

EPSILON_ZERO = 1e-10
EPSILON_LOG = 1e-12


def build_embedded_vectors(
    signal: NDArray[np.float64], 
    embedding_dim: int, 
    time_delay: int
) -> NDArray[np.float64]:
    n_vectors = len(signal) - (embedding_dim - 1) * time_delay
    if n_vectors <= 0:
        raise ValueError(
            f"Signal length {len(signal)} too short for m={embedding_dim}, tau={time_delay}"
        )
    
    embedded = np.zeros((n_vectors, embedding_dim))
    for j in range(embedding_dim):
        embedded[:, j] = signal[j * time_delay:j * time_delay + n_vectors]
    
    return embedded


def compute_ordinal_patterns(
    embedded_vectors: NDArray[np.float64]
) -> NDArray[np.int_]:
    return np.argsort(embedded_vectors, axis=1)


def build_probability_distribution(
    ordinal_patterns: NDArray[np.int_],
    embedding_dim: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    patterns_as_tuples = [tuple(row) for row in ordinal_patterns]
    pattern_counts = Counter(patterns_as_tuples)
    total_patterns = len(ordinal_patterns)
    total_patterns_safe = total_patterns + EPSILON_ZERO
    
    observed_prob = np.array([
        count / total_patterns_safe for count in pattern_counts.values()
    ])
    
    all_patterns = list(permutations(range(embedding_dim)))
    n_permutations = len(all_patterns)
    
    full_prob = np.array([
        pattern_counts.get(pattern, 0) / total_patterns_safe
        for pattern in all_patterns
    ])
    
    return observed_prob, full_prob, n_permutations


def compute_shannon_entropy(distribution: NDArray) -> float:
    return float(-np.sum(distribution * np.log(distribution + EPSILON_LOG)))


def compute_js_divergence(
    prob_full: NDArray[np.float64],
    prob_uniform: NDArray[np.float64],
    entropy_p: float,
    entropy_uniform: float
) -> float:
    mixture = ((prob_full + prob_uniform) / 2).astype(np.float64)
    entropy_mixture = compute_shannon_entropy(mixture)
    return float(entropy_mixture - 0.5 * entropy_p - 0.5 * entropy_uniform)


def compute_max_divergence(
    prob_uniform: NDArray[np.float64],
    entropy_uniform: float
) -> float:
    prob_delta = np.zeros(len(prob_uniform))
    prob_delta[0] = 1.0
    
    mixture_max = ((prob_delta + prob_uniform) / 2).astype(np.float64)
    entropy_mixture_max = compute_shannon_entropy(mixture_max)
    
    return float(entropy_mixture_max - 0.5 * entropy_uniform)