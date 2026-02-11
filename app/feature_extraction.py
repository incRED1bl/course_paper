import numpy as np
from .data_preprocessing import extract_frequency_features, detect_whistles
from .func import compute_entropy_complexity


def extract_all_features(signals, sample_rate, m=3, tau=1):
    features_dict = {}
    whistle_results = {}
    entropy_complexity = {}
    
    for disease_name, signal_data in signals.items():
        frequencies, magnitudes, feature_vector = extract_frequency_features(signal_data, sample_rate)
        features_dict[disease_name] = {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'features': feature_vector
        }
        
        whistle_detected, whistle_strength = detect_whistles(signal_data, sample_rate)
        whistle_results[disease_name] = {
            'detected': whistle_detected,
            'strength': whistle_strength
        }
        
        entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
        entropy_complexity[disease_name] = {
            'entropy': entropy,
            'complexity': complexity
        }
    
    return features_dict, whistle_results, entropy_complexity


def build_feature_summary(signals, features_dict, whistle_results, entropy_complexity):
    disease_features = {}
    
    for disease_name in signals.keys():
        freq_features = features_dict[disease_name]['features']
        whistle = whistle_results[disease_name]
        entropy_comp = entropy_complexity[disease_name]
        
        total_energy = (freq_features['low_freq_energy'] + 
                       freq_features['mid_freq_energy'] + 
                       freq_features['high_freq_energy'])
        
        disease_features[disease_name] = {
            'Low Freq Energy': freq_features['low_freq_energy'] / total_energy,
            'Mid Freq Energy': freq_features['mid_freq_energy'] / total_energy,
            'High Freq Energy': freq_features['high_freq_energy'] / total_energy,
            'Whistle Strength': whistle['strength'],
            'Spectral Centroid': freq_features['spectral_centroid'],
            'Peak Frequency': freq_features['peak_frequency'],
            'Entropy': entropy_comp['entropy'],
            'Complexity': entropy_comp['complexity']
        }
    
    return disease_features


def print_feature_summary(disease_features):
    print("Feature Summary:")
    print("-" * 80)
    for disease, features in disease_features.items():
        print(f"\n{disease}:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value:.4f}")
