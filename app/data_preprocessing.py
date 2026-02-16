"""
Data preprocessing functions for working with audio files and creating DataFrames.
Handles batch processing, DataFrame creation, and data organization.
"""
import numpy as np
from .features import (
    compute_entropy_complexity,
    extract_frequency_features,
    detect_whistles
)


def extract_features_row(filename, signal_data, sample_rate, m=3, tau=1):
    """
    Extract all features from a single audio signal and return as a dictionary ready for DataFrame.
    
    Args:
        filename: Audio file name
        signal_data: Audio signal array
        sample_rate: Sample rate of the audio
        m: Embedding dimension for entropy-complexity (default: 3)
        tau: Time delay for entropy-complexity (default: 1)
    
    Returns:
        Dictionary with all extracted features (ready for DataFrame row)
    """
    frequencies, magnitudes, freq_features = extract_frequency_features(signal_data, sample_rate)
    
    whistle_detected, whistle_strength = detect_whistles(signal_data, sample_rate)
    
    entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
    
    total_energy = (freq_features['low_freq_energy'] + 
                   freq_features['mid_freq_energy'] + 
                   freq_features['high_freq_energy'])
    
    row = {
        'filename': filename,
        'patient_id': int(filename.split('_')[0]),
        'low_freq_energy': freq_features['low_freq_energy'] / total_energy,
        'mid_freq_energy': freq_features['mid_freq_energy'] / total_energy,
        'high_freq_energy': freq_features['high_freq_energy'] / total_energy,
        'whistle_strength': whistle_strength,
        'spectral_centroid': freq_features['spectral_centroid'],
        'peak_frequency': freq_features['peak_frequency'],
        'entropy': entropy,
        'complexity': complexity
    }
    
    return row


def extract_features_batch(signals_dict, m=3, tau=1):
    """
    Extract features from multiple audio signals and return list of rows.
    
    Args:
        signals_dict: Dictionary where keys are filenames and values are dicts with 'signal' and 'sample_rate'
        m: Embedding dimension for entropy-complexity (default: 3)
        tau: Time delay for entropy-complexity (default: 1)
    
    Returns:
        List of dictionaries, each containing features for one audio file
    """
    rows = []
    
    for filename, data in signals_dict.items():
        signal_data = data['signal']
        sample_rate = data['sample_rate']
        
        row = extract_features_row(filename, signal_data, sample_rate, m=m, tau=tau)
        rows.append(row)
    
    return rows


def extract_all_features(signals, sample_rate, m=3, tau=1):
    """
    Extract all feature types for multiple signals and organize into dictionaries.
    
    Args:
        signals: Dictionary where keys are signal names and values are signal arrays
        sample_rate: Sample rate of the audio signals
        m: Embedding dimension for entropy-complexity (default: 3)
        tau: Time delay for entropy-complexity (default: 1)
    
    Returns:
        Tuple of (features_dict, whistle_results, entropy_complexity)
    """
    features_dict = {}
    whistle_results = {}
    entropy_complexity = {}
    
    for signal_name, signal_data in signals.items():
        frequencies, magnitudes, feature_vector = extract_frequency_features(signal_data, sample_rate)
        features_dict[signal_name] = {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'features': feature_vector
        }
        
        whistle_detected, whistle_strength = detect_whistles(signal_data, sample_rate)
        whistle_results[signal_name] = {
            'detected': whistle_detected,
            'strength': whistle_strength
        }
        
        entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
        entropy_complexity[signal_name] = {
            'entropy': entropy,
            'complexity': complexity
        }
    
    return features_dict, whistle_results, entropy_complexity


def build_feature_summary(signals, features_dict, whistle_results, entropy_complexity):
    """
    Build a comprehensive feature summary dictionary for all signals.
    
    Args:
        signals: Dictionary of signal names to signal arrays
        features_dict: Dictionary of frequency features
        whistle_results: Dictionary of whistle detection results
        entropy_complexity: Dictionary of entropy-complexity values
    
    Returns:
        Dictionary mapping signal names to normalized feature dictionaries
    """
    signal_features = {}
    
    for signal_name in signals.keys():
        freq_features = features_dict[signal_name]['features']
        whistle = whistle_results[signal_name]
        entropy_comp = entropy_complexity[signal_name]
        
        total_energy = (freq_features['low_freq_energy'] + 
                       freq_features['mid_freq_energy'] + 
                       freq_features['high_freq_energy'])
        
        signal_features[signal_name] = {
            'Low Freq Energy': freq_features['low_freq_energy'] / total_energy,
            'Mid Freq Energy': freq_features['mid_freq_energy'] / total_energy,
            'High Freq Energy': freq_features['high_freq_energy'] / total_energy,
            'Whistle Strength': whistle['strength'],
            'Spectral Centroid': freq_features['spectral_centroid'],
            'Peak Frequency': freq_features['peak_frequency'],
            'Entropy': entropy_comp['entropy'],
            'Complexity': entropy_comp['complexity']
        }
    
    return signal_features


def print_feature_summary(signal_features):
    """
    Print a formatted summary of all extracted features.
    
    Args:
        signal_features: Dictionary mapping signal names to feature dictionaries
    """
    print("Feature Summary:")
    print("-" * 80)
    for signal_name, features in signal_features.items():
        print(f"\n{signal_name}:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value:.4f}")
