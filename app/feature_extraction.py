import numpy as np
from scipy import signal
from .entropy_complexity import compute_entropy_complexity


def extract_frequency_features(audio_data, sample_rate, n_fft=2048):
    """Extract frequency domain features."""
    fft_result = np.fft.rfft(audio_data, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1/sample_rate)
    magnitudes = np.abs(fft_result)
    
    feature_vector = {
        'spectral_centroid': np.sum(frequencies * magnitudes) / np.sum(magnitudes),
        'spectral_bandwidth': np.sqrt(np.sum(((frequencies - np.sum(frequencies * magnitudes) / np.sum(magnitudes)) ** 2) * magnitudes) / np.sum(magnitudes)),
        'spectral_rolloff': frequencies[np.where(np.cumsum(magnitudes) >= 0.85 * np.sum(magnitudes))[0][0]],
        'low_freq_energy': np.sum(magnitudes[frequencies < 500]),
        'mid_freq_energy': np.sum(magnitudes[(frequencies >= 500) & (frequencies < 1000)]),
        'high_freq_energy': np.sum(magnitudes[frequencies >= 1000]),
        'peak_frequency': frequencies[np.argmax(magnitudes)]
    }
    
    return frequencies, magnitudes, feature_vector


def detect_whistles(audio_data, sample_rate, whistle_freq_range=(400, 1600)):
    """Detect whistle sounds in audio."""
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=256)
    
    freq_mask = (f >= whistle_freq_range[0]) & (f <= whistle_freq_range[1])
    whistle_energy = np.sum(Sxx[freq_mask, :])
    total_energy = np.sum(Sxx)
    
    whistle_strength = whistle_energy / total_energy if total_energy > 0 else 0
    whistle_detected = whistle_strength > 0.2
    
    return whistle_detected, whistle_strength


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
