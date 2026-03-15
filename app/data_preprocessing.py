from typing import Any

import numpy as np
from numpy.typing import NDArray

from .features import (
    AudioFeatureExtractor,
    compute_entropy_complexity,
    extract_frequency_features,
)

EPSILON_ZERO = 1e-10


def extract_features_row(
    filename: str,
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    extractor: AudioFeatureExtractor | None = None,
    embedding_dim: int = 3,
    time_delay: int = 1
) -> dict[str, Any]:
    if extractor is None:
        extractor = AudioFeatureExtractor()

    _, _, freq_features = extract_frequency_features(
        audio_signal, sample_rate
    )
    
    entropy, complexity = compute_entropy_complexity(
        audio_signal, embedding_dim=embedding_dim, time_delay=time_delay
    )
    
    total_energy = (
        freq_features['low_freq_energy'] + 
        freq_features['mid_freq_energy'] + 
        freq_features['high_freq_energy']
    )
    total_energy_safe = total_energy + EPSILON_ZERO

    advanced_features = extractor.extract_golden(audio_signal, sample_rate)
    
    return {
        'filename': filename,
        'patient_id': int(filename.split('_')[0]),
        'low_freq_energy': freq_features['low_freq_energy'] / total_energy_safe,
        'spectral_centroid': freq_features['spectral_centroid'],
        'entropy': entropy,
        'complexity': complexity,
        **advanced_features,
    }


def extract_features_batch(
    signals_dict: dict[str, dict[str, Any]],
    extractor: AudioFeatureExtractor | None = None,
    embedding_dim: int = 3,
    time_delay: int = 1
) -> list[dict[str, Any]]:
    if extractor is None:
        extractor = AudioFeatureExtractor()

    return [
        extract_features_row(
            filename, 
            data['signal'], 
            data['sample_rate'],
            extractor=extractor,
            embedding_dim=embedding_dim, 
            time_delay=time_delay
        )
        for filename, data in signals_dict.items()
    ]

