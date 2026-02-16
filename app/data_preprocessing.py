"""
Data preprocessing and batch feature extraction for audio signals.

This module provides functions for:
- Single signal feature extraction
- Batch processing of multiple audio files
- DataFrame-ready feature dictionaries
- Feature summary generation and display
"""
from typing import Dict, List, Any, Tuple

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
) -> Dict[str, Any]:
    """
    Extract all features from a single audio signal for DataFrame insertion.
    
    Computes frequency features, whistle detection, and entropy-complexity metrics,
    returning a dictionary suitable for pandas DataFrame row insertion.
    
    Args:
        filename: Audio file name (must contain patient ID as prefix: '<id>_...')
        audio_signal: Input audio signal as 1D array
        sample_rate: Sampling rate of the audio in Hz
        embedding_dim: Embedding dimension for entropy-complexity computation. Default is 3.
        time_delay: Time delay for entropy-complexity computation. Default is 1.
    
    Returns:
        Dictionary with keys:
            - 'filename': Original filename
            - 'patient_id': Extracted patient ID from filename
            - 'low_freq_energy': Normalized low frequency energy [0,1]
            - 'mid_freq_energy': Normalized mid frequency energy [0,1]
            - 'high_freq_energy': Normalized high frequency energy [0,1]
            - 'whistle_strength': Whistle detection strength [0,1]
            - 'spectral_centroid': Spectral centroid in Hz
            - 'peak_frequency': Peak frequency in Hz
            - 'entropy': Normalized permutation entropy [0,1]
            - 'complexity': Statistical complexity [0,1]
    
    Raises:
        ValueError: If filename doesn't contain valid patient ID format.
    """
    frequencies, magnitudes, freq_features = extract_frequency_features(
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
    
    row = {
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
    
    return row


def extract_features_batch(
    signals_dict: Dict[str, Dict[str, Any]],
    embedding_dim: int = 3,
    time_delay: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract features from multiple audio signals in batch.
    
    Processes a dictionary of audio signals and returns a list of feature dictionaries,
    each representing one audio file's extracted features.
    
    Args:
        signals_dict: Dictionary mapping filenames to signal data:
            - keys: Filenames (str)
            - values: Dict with 'signal' (NDArray) and 'sample_rate' (int)
        embedding_dim: Embedding dimension for entropy-complexity computation. Default is 3.
        time_delay: Time delay for entropy-complexity computation. Default is 1.
    
    Returns:
        List of feature dictionaries, one per input signal. Each dictionary has the
        same structure as returned by extract_features_row().
    
    Example:
        >>> signals = {
        ...     '101_1b1_Al_sc_Meditron.wav': {
        ...         'signal': audio_array,
        ...         'sample_rate': 4000
        ...     }
        ... }
        >>> rows = extract_features_batch(signals)
    """
    rows = []
    
    for filename, data in signals_dict.items():
        audio_signal = data['signal']
        sample_rate = data['sample_rate']
        
        row = extract_features_row(
            filename, audio_signal, sample_rate,
            embedding_dim=embedding_dim, time_delay=time_delay
        )
        rows.append(row)
    
    return rows


def extract_all_features(
    signals: Dict[str, NDArray[np.float64]],
    sample_rate: int,
    embedding_dim: int = 3,
    time_delay: int = 1
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Extract all feature types for multiple signals and organize into separate dictionaries.
    
    Processes each signal to compute frequency features, whistle detection, and
    entropy-complexity metrics, organizing results by feature type.
    
    Args:
        signals: Dictionary mapping signal names to audio signal arrays
        sample_rate: Sampling rate of all audio signals in Hz
        embedding_dim: Embedding dimension for entropy-complexity computation. Default is 3.
        time_delay: Time delay for entropy-complexity computation. Default is 1.
    
    Returns:
        Tuple containing three dictionaries:
            1. features_dict: Frequency domain features
                Format: {signal_name: {'frequencies': array, 'magnitudes': array, 'features': dict}}
            2. whistle_results: Whistle detection results
                Format: {signal_name: {'detected': bool, 'strength': float}}
            3. entropy_complexity: Entropy and complexity metrics
                Format: {signal_name: {'entropy': float, 'complexity': float}}
    
    Example:
        >>> signals = {'signal1': audio_array1, 'signal2': audio_array2}
        >>> freq_feats, whistles, entropy_comp = extract_all_features(signals, 4000)
    """
    features_dict = {}
    whistle_results = {}
    entropy_complexity = {}
    
    for signal_name, audio_signal in signals.items():
        frequencies, magnitudes, feature_vector = extract_frequency_features(
            audio_signal, sample_rate
        )
        features_dict[signal_name] = {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'features': feature_vector,
        }
        
        whistle_detected, whistle_strength = detect_whistles(audio_signal, sample_rate)
        whistle_results[signal_name] = {
            'detected': whistle_detected,
            'strength': whistle_strength,
        }
        
        entropy, complexity = compute_entropy_complexity(
            audio_signal, embedding_dim=embedding_dim, time_delay=time_delay
        )
        entropy_complexity[signal_name] = {
            'entropy': entropy,
            'complexity': complexity,
        }
    
    return features_dict, whistle_results, entropy_complexity


def build_feature_summary(
    signals: Dict[str, NDArray[np.float64]],
    features_dict: Dict[str, Dict[str, Any]],
    whistle_results: Dict[str, Dict[str, Any]],
    entropy_complexity: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Build a comprehensive feature summary with normalized energy features.
    
    Combines all extracted features into a single summary dictionary with
    human-readable feature names and normalized energy values.
    
    Args:
        signals: Dictionary mapping signal names to audio signal arrays
        features_dict: Frequency domain features from extract_all_features()
        whistle_results: Whistle detection results from extract_all_features()
        entropy_complexity: Entropy-complexity metrics from extract_all_features()
    
    Returns:
        Dictionary mapping signal names to feature dictionaries with keys:
            - 'Low Freq Energy': Normalized low frequency energy [0,1]
            - 'Mid Freq Energy': Normalized mid frequency energy [0,1]
            - 'High Freq Energy': Normalized high frequency energy [0,1]
            - 'Whistle Strength': Whistle detection strength [0,1]
            - 'Spectral Centroid': Spectral centroid in Hz
            - 'Peak Frequency': Peak frequency in Hz
            - 'Entropy': Normalized permutation entropy [0,1]
            - 'Complexity': Statistical complexity [0,1]
    
    Example:
        >>> summary = build_feature_summary(signals, freq_dict, whistle_dict, entropy_dict)
        >>> print(summary['signal1']['Entropy'])
        0.8234
    """
    signal_features = {}
    
    for signal_name in signals.keys():
        freq_features = features_dict[signal_name]['features']
        whistle = whistle_results[signal_name]
        entropy_comp = entropy_complexity[signal_name]
        
        total_energy = (
            freq_features['low_freq_energy'] + 
            freq_features['mid_freq_energy'] + 
            freq_features['high_freq_energy']
        )
        total_energy_safe = total_energy + EPSILON_ZERO
        
        signal_features[signal_name] = {
            'Low Freq Energy': freq_features['low_freq_energy'] / total_energy_safe,
            'Mid Freq Energy': freq_features['mid_freq_energy'] / total_energy_safe,
            'High Freq Energy': freq_features['high_freq_energy'] / total_energy_safe,
            'Whistle Strength': whistle['strength'],
            'Spectral Centroid': freq_features['spectral_centroid'],
            'Peak Frequency': freq_features['peak_frequency'],
            'Entropy': entropy_comp['entropy'],
            'Complexity': entropy_comp['complexity'],
        }
    
    return signal_features


def print_feature_summary(signal_features: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted summary of all extracted features to console.
    
    Displays features in a human-readable format with consistent decimal precision.
    
    Args:
        signal_features: Dictionary from build_feature_summary() mapping signal names
            to feature dictionaries.
    
    Example:
        >>> print_feature_summary(summary)
        Feature Summary:
        --------------------------------------------------------------------------------
        
        signal1:
          Low Freq Energy: 0.4523
          Mid Freq Energy: 0.3211
          ...
    """
    print("Feature Summary:")
    print("-" * 80)
    for signal_name, features in signal_features.items():
        print(f"\n{signal_name}:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value:.4f}")
