"""
Audio signal analysis package.
Provides functions for feature extraction from respiratory sounds.
"""

# Feature computation functions (high-level)
from .features import (
    compute_entropy_complexity,
    extract_frequency_features,
    detect_whistles
)

# Data preprocessing functions (working with files/DataFrames)
from .data_preprocessing import (
    extract_features_row,
    extract_features_batch,
    extract_all_features,
    build_feature_summary,
    print_feature_summary
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
    'print_feature_summary'
]
