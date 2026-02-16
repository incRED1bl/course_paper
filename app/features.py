"""
Feature extraction functions for audio signal analysis.

This module provides high-level feature extraction functions for respiratory sound analysis:
- Permutation entropy and statistical complexity
- Frequency domain features
- Whistle/wheeze detection
"""
import math
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .computations import (
    build_embedded_vectors,
    compute_ordinal_patterns,
    build_probability_distribution,
    compute_shannon_entropy,
    compute_js_divergence,
    compute_max_divergence,
    EPSILON_ZERO,
)


def compute_entropy_complexity(
    time_series: NDArray[np.float64],
    embedding_dim: int = 3,
    time_delay: int = 1
) -> Tuple[float, float]:
    """
    Compute permutation entropy and statistical complexity using the Bandt-Pompe method.
    
    Uses ordinal patterns to quantify randomness (entropy) and structure (complexity)
    in time series data. Normalized entropy H ∈ [0,1] and complexity C ∈ [0,1].
    
    Args:
        time_series: Input time series signal as 1D array.
        embedding_dim: Embedding dimension for phase space reconstruction. Default is 3.
        time_delay: Time delay between consecutive elements in embedding vectors. Default is 1.
    
    Returns:
        Tuple containing:
            - normalized_entropy (float): Shannon entropy normalized by maximum entropy H ∈ [0,1]
            - complexity (float): Statistical complexity measure C ∈ [0,1]
    
    Raises:
        ValueError: If time_series is empty or embedding_dim < 2.
    
    References:
        Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure
        for time series. Physical Review Letters, 88(17), 174102.
    """
    signal_array = np.array(time_series)
    
    embedded_vectors = build_embedded_vectors(signal_array, embedding_dim, time_delay)
    ordinal_patterns = compute_ordinal_patterns(embedded_vectors)
    prob_dist, prob_full, n_permutations = build_probability_distribution(
        ordinal_patterns, embedding_dim
    )
    
    entropy = compute_shannon_entropy(prob_dist)
    max_entropy = np.log(math.factorial(embedding_dim))
    normalized_entropy = entropy / max_entropy
    
    uniform_dist = np.ones(n_permutations) / n_permutations
    uniform_entropy = compute_shannon_entropy(uniform_dist)
    
    js_div = compute_js_divergence(prob_full, uniform_dist, entropy, uniform_entropy)
    max_js_div = compute_max_divergence(uniform_dist, uniform_entropy)
    
    complexity = (js_div / max_js_div) * normalized_entropy
    
    return normalized_entropy, complexity


def extract_frequency_features(
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    n_fft: int = 2048
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, float]]:
    """
    Extract frequency domain features from audio signal using Fast Fourier Transform.
    
    Computes spectral features including centroid, bandwidth, rolloff, and energy distribution
    across frequency bands. Useful for characterizing respiratory sounds.
    
    Args:
        audio_signal: Input audio signal as 1D array.
        sample_rate: Sampling rate of the audio in Hz.
        n_fft: FFT window size (number of points). Default is 2048.
    
    Returns:
        Tuple containing:
            - frequencies (NDArray): Frequency bins in Hz
            - magnitudes (NDArray): Magnitude spectrum
            - features (Dict[str, float]): Dictionary with keys:
                - 'spectral_centroid': Center of mass of spectrum (Hz)
                - 'spectral_bandwidth': Standard deviation of spectrum (Hz)
                - 'spectral_rolloff': Frequency below which 85% of energy exists (Hz)
                - 'low_freq_energy': Energy in [0, 500) Hz range
                - 'mid_freq_energy': Energy in [500, 1000) Hz range
                - 'high_freq_energy': Energy in [1000, ∞) Hz range
                - 'peak_frequency': Frequency with maximum magnitude (Hz)
    
    Raises:
        ValueError: If audio_signal is empty or sample_rate <= 0.
    """
    fft_result = np.fft.rfft(audio_signal, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    magnitudes = np.abs(fft_result)
    
    total_magnitude = np.sum(magnitudes) + EPSILON_ZERO
    spectral_centroid = np.sum(frequencies * magnitudes) / total_magnitude
    
    spectral_variance = np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / total_magnitude
    spectral_bandwidth = np.sqrt(spectral_variance)
    
    cumulative_energy = np.cumsum(magnitudes)
    rolloff_threshold = 0.85 * np.sum(magnitudes)
    rolloff_index = np.where(cumulative_energy >= rolloff_threshold)[0][0]
    spectral_rolloff = frequencies[rolloff_index]
    
    features = {
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'low_freq_energy': float(np.sum(magnitudes[frequencies < 500])),
        'mid_freq_energy': float(np.sum(magnitudes[(frequencies >= 500) & (frequencies < 1000)])),
        'high_freq_energy': float(np.sum(magnitudes[frequencies >= 1000])),
        'peak_frequency': float(frequencies[np.argmax(magnitudes)]),
    }
    
    return frequencies, magnitudes, features


def detect_whistles(
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    freq_range: Tuple[int, int] = (400, 1600)
) -> Tuple[bool, float]:
    """
    Detect whistle sounds (wheezes) in respiratory audio using spectrogram analysis.
    
    Computes time-frequency representation and measures energy concentration
    in the typical wheeze frequency range. Wheezes are continuous musical sounds
    with dominant frequencies typically in 400-1600 Hz range.
    
    Args:
        audio_signal: Input audio signal as 1D array.
        sample_rate: Sampling rate of the audio in Hz.
        freq_range: Tuple (low_freq, high_freq) defining whistle frequency band in Hz.
            Default is (400, 1600) Hz based on respiratory sound literature.
    
    Returns:
        Tuple containing:
            - whistle_detected (bool): True if whistle strength exceeds threshold (0.2)
            - whistle_strength (float): Ratio of energy in whistle band to total energy [0,1]
    
    Raises:
        ValueError: If audio_signal is empty, sample_rate <= 0, or invalid freq_range.
    
    Notes:
        Detection threshold of 0.2 (20% energy in whistle band) is empirically determined
        for respiratory sound analysis.
    """
    frequencies, times, spectrogram = signal.spectrogram(
        audio_signal, sample_rate, nperseg=256
    )
    
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    whistle_band_energy = np.sum(spectrogram[freq_mask, :])
    total_energy = np.sum(spectrogram) + EPSILON_ZERO
    
    whistle_strength = whistle_band_energy / total_energy
    whistle_detected = whistle_strength > 0.2
    
    return whistle_detected, float(whistle_strength)
