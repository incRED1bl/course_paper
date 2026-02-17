import math

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
) -> tuple[float, float]:
    embedded_vectors = build_embedded_vectors(time_series, embedding_dim, time_delay)
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    fft_result = np.fft.rfft(audio_signal, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate).astype(np.float64)
    magnitudes = np.abs(fft_result).astype(np.float64)
    
    total_magnitude = np.sum(magnitudes) + EPSILON_ZERO
    spectral_centroid = np.sum(frequencies * magnitudes) / total_magnitude
    
    spectral_variance = np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / total_magnitude
    spectral_bandwidth = np.sqrt(spectral_variance)
    
    cumulative_energy = np.cumsum(magnitudes)
    rolloff_threshold = 0.85 * cumulative_energy[-1]
    rolloff_index = np.searchsorted(cumulative_energy, rolloff_threshold)
    spectral_rolloff = frequencies[rolloff_index]
    
    freq_bins = np.digitize(frequencies, [500, 1000])
    low_freq_energy = float(np.sum(magnitudes[freq_bins == 0]))
    mid_freq_energy = float(np.sum(magnitudes[freq_bins == 1]))
    high_freq_energy = float(np.sum(magnitudes[freq_bins == 2]))
    
    features = {
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'low_freq_energy': low_freq_energy,
        'mid_freq_energy': mid_freq_energy,
        'high_freq_energy': high_freq_energy,
        'peak_frequency': float(frequencies[np.argmax(magnitudes)]),
    }
    
    return frequencies, magnitudes, features


def detect_whistles(
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    freq_range: tuple[int, int] = (400, 1600)
) -> tuple[bool, float]:
    frequencies, times, spectrogram = signal.spectrogram(
        audio_signal, sample_rate, nperseg=256
    )
    
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    whistle_band_energy = np.sum(spectrogram[freq_mask, :])
    total_energy = np.sum(spectrogram) + EPSILON_ZERO
    
    whistle_strength = whistle_band_energy / total_energy
    whistle_detected = whistle_strength > 0.2
    
    return whistle_detected, float(whistle_strength)
