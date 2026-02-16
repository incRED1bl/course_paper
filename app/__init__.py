"""
Audio signal analysis package for respiratory sound feature extraction.

This package implements the Bandt-Pompe permutation entropy method and
frequency-domain analysis for respiratory sound classification.

Modules:
    computations: Low-level mathematical helper functions (entropy, divergence)
    features: High-level feature extraction (entropy-complexity, frequency, whistles)
    data_preprocessing: Batch processing and DataFrame utilities

Main Features:
    - Permutation entropy and statistical complexity computation
    - FFT-based spectral feature extraction
    - Whistle/wheeze detection using spectrograms
    - Batch processing for multiple audio files
    - DataFrame-ready feature dictionaries

Example:
    >>> from app import compute_entropy_complexity, extract_frequency_features
    >>> entropy, complexity = compute_entropy_complexity(signal, embedding_dim=3)
    >>> freqs, mags, features = extract_frequency_features(signal, sample_rate=4000)
"""

# Feature computation functions (high-level)
from .features import (
    compute_entropy_complexity,
    extract_frequency_features,
    detect_whistles,
)

# Data preprocessing functions (working with files/DataFrames)
from .data_preprocessing import (
    extract_features_row,
    extract_features_batch,
    extract_all_features,
    build_feature_summary,
    print_feature_summary,
)

__all__ = [
    # Feature functions
    'compute_entropy_complexity',
    'extract_frequency_features',
    'detect_whistles',
    # Data preprocessing functions
    'extract_features_row',
    'extract_features_batch',
    'extract_all_features',
    'build_feature_summary',
    'print_feature_summary',
]
