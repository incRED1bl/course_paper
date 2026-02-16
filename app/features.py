"""
Feature extraction functions for audio signals.
Each function computes a specific type of feature (entropy, complexity, frequency, whistles).
"""
import numpy as np
from .computations import (
    _construct_embedded_vectors,
    _compute_ordinal_patterns,
    _build_probability_distributions,
    _compute_shannon_entropy,
    _compute_jensen_shannon_divergence,
    _compute_max_divergence
)
from scipy import signal
import math


def compute_entropy_complexity(y, m=3, tau=1):
    """
    Compute permutation entropy and statistical complexity using Bandt-Pompe method.
    
    Args:
        y: Time series signal
        m: Embedding dimension (default: 3)
        tau: Time delay (default: 1)
    
    Returns:
        Tuple of (normalized_entropy, complexity)
    """
    y = np.array(y)
    
    embedded_vectors = _construct_embedded_vectors(y, m, tau)
    ordinal_patterns = _compute_ordinal_patterns(embedded_vectors)
    P, P_full, n_permutations = _build_probability_distributions(ordinal_patterns, m)
    
    S_P = _compute_shannon_entropy(P)
    S_max = np.log(math.factorial(m))
    H_P = S_P / S_max
    
    P_u = np.ones(n_permutations) / n_permutations
    S_Pu = _compute_shannon_entropy(P_u)
    
    J = _compute_jensen_shannon_divergence(P_full, P_u, S_P, S_Pu)
    J_max = _compute_max_divergence(P_u, S_Pu)
    
    C_P = (J / J_max) * H_P
    
    return H_P, C_P


def extract_frequency_features(audio_data, sample_rate, n_fft=2048):
    """
    Extract frequency domain features from audio signal using FFT.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Sample rate of the audio
        n_fft: FFT window size (default: 2048)
    
    Returns:
        Tuple of (frequencies, magnitudes, feature_dict)
        where feature_dict contains:
            - spectral_centroid: Center of mass of the spectrum
            - spectral_bandwidth: Width of the spectrum
            - spectral_rolloff: Frequency below which 85% of energy is contained
            - low_freq_energy: Energy in low frequencies (<500 Hz)
            - mid_freq_energy: Energy in mid frequencies (500-1000 Hz)
            - high_freq_energy: Energy in high frequencies (>1000 Hz)
            - peak_frequency: Frequency with maximum magnitude
    """
    fft_result = np.fft.rfft(audio_data, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1/sample_rate)
    magnitudes = np.abs(fft_result)
    
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    
    feature_vector = {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / np.sum(magnitudes)),
        'spectral_rolloff': frequencies[np.where(np.cumsum(magnitudes) >= 0.85 * np.sum(magnitudes))[0][0]],
        'low_freq_energy': np.sum(magnitudes[frequencies < 500]),
        'mid_freq_energy': np.sum(magnitudes[(frequencies >= 500) & (frequencies < 1000)]),
        'high_freq_energy': np.sum(magnitudes[frequencies >= 1000]),
        'peak_frequency': frequencies[np.argmax(magnitudes)]
    }
    
    return frequencies, magnitudes, feature_vector


def detect_whistles(audio_data, sample_rate, whistle_freq_range=(400, 1600)):
    """
    Detect whistle sounds (wheezes) in audio signal using spectrogram analysis.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Sample rate of the audio
        whistle_freq_range: Frequency range for whistle detection (default: 400-1600 Hz)
    
    Returns:
        Tuple of (whistle_detected (bool), whistle_strength (float))
        where whistle_strength is the ratio of energy in whistle range to total energy
    """
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=256)
    
    freq_mask = (f >= whistle_freq_range[0]) & (f <= whistle_freq_range[1])
    whistle_energy = np.sum(Sxx[freq_mask, :])
    total_energy = np.sum(Sxx)
    
    whistle_strength = whistle_energy / total_energy if total_energy > 0 else 0
    whistle_detected = whistle_strength > 0.2
    
    return whistle_detected, whistle_strength
