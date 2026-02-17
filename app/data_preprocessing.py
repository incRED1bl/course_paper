from typing import Any

import numpy as np
from numpy.typing import NDArray

from .computations import EPSILON_ZERO
from .features import (
    compute_entropy_complexity,
    extract_frequency_features,
    detect_whistles,
)


def extract_features_row(
    filename: str,
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    embedding_dim: int = 3,
    time_delay: int = 1
) -> dict[str, Any]:
    _, _, freq_features = extract_frequency_features(
        audio_signal, sample_rate
    )
    
    whistle_detected, whistle_strength = detect_whistles(audio_signal, sample_rate)
    
    entropy, complexity = compute_entropy_complexity(
        audio_signal, embedding_dim=embedding_dim, time_delay=time_delay
    )
    
    total_energy = (
        freq_features['low_freq_energy'] + 
        freq_features['mid_freq_energy'] + 
        freq_features['high_freq_energy']
    )
    total_energy_safe = total_energy + EPSILON_ZERO
    
    return {
        'filename': filename,
        'patient_id': int(filename.split('_')[0]),
        'low_freq_energy': freq_features['low_freq_energy'] / total_energy_safe,
        'mid_freq_energy': freq_features['mid_freq_energy'] / total_energy_safe,
        'high_freq_energy': freq_features['high_freq_energy'] / total_energy_safe,
        'whistle_strength': whistle_strength,
        'spectral_centroid': freq_features['spectral_centroid'],
        'peak_frequency': freq_features['peak_frequency'],
        'entropy': entropy,
        'complexity': complexity,
    }


def extract_features_batch(
    signals_dict: dict[str, dict[str, Any]],
    embedding_dim: int = 3,
    time_delay: int = 1
) -> list[dict[str, Any]]:
    return [
        extract_features_row(
            filename, 
            data['signal'], 
            data['sample_rate'],
            embedding_dim=embedding_dim, 
            time_delay=time_delay
        )
        for filename, data in signals_dict.items()
    ]

