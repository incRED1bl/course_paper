# API Documentation

## Package: `app`

Audio signal analysis package for respiratory sound feature extraction.

This package implements the Bandt-Pompe permutation entropy method and frequency-domain analysis for respiratory sound classification.

**Modules:**
- `computations` — Low-level mathematical helper functions (entropy, divergence)
- `features` — High-level feature extraction (entropy-complexity, frequency, whistles)
- `data_preprocessing` — Batch processing and DataFrame utilities

**Main Features:**
- Permutation entropy and statistical complexity computation
- FFT-based spectral feature extraction
- Whistle/wheeze detection using spectrograms
- Batch processing for multiple audio files
- DataFrame-ready feature dictionaries

**Example:**
```python
from app import compute_entropy_complexity, extract_frequency_features
entropy, complexity = compute_entropy_complexity(signal, embedding_dim=3)
freqs, mags, features = extract_frequency_features(signal, sample_rate=4000)
```

---

## Module: `app.computations`

Low-level mathematical functions for time-series feature computation.

### Constants

- **`EPSILON_ZERO`** = `1e-10` — Small value to prevent division by zero
- **`EPSILON_LOG`** = `1e-12` — Small value for logarithm stability

---

### `build_embedded_vectors(signal, embedding_dim, time_delay)`

Create time-delay embedded vectors from signal.

**Parameters:**
- `signal: NDArray[np.float64]` — Input time series
- `embedding_dim: int` — Embedding dimension (m)
- `time_delay: int` — Time delay between samples (tau)

**Returns:**
- `list[list[float]]` — List of embedded vectors

**Raises:**
- `ValueError` — If signal is too short for given parameters

---

### `compute_ordinal_patterns(embedded_vectors)`

Convert embedded vectors to ordinal patterns.

**Parameters:**
- `embedded_vectors: list[list[float]]` — List of embedded vectors

**Returns:**
- `list[tuple[int, ...]]` — List of ordinal patterns as tuples

---

### `build_probability_distribution(ordinal_patterns, embedding_dim)`

Build probability distributions from ordinal patterns.

**Parameters:**
- `ordinal_patterns: list[tuple[int, ...]]` — List of ordinal patterns
- `embedding_dim: int` — Embedding dimension

**Returns:**
- `tuple[NDArray[np.float64], NDArray[np.float64], int]` — Tuple of (observed_prob, full_prob, n_permutations)

---

### `compute_shannon_entropy(distribution)`

Calculate Shannon entropy of probability distribution.

**Parameters:**
- `distribution: NDArray` — Probability distribution

**Returns:**
- `float` — Shannon entropy value

---

### `compute_js_divergence(prob_full, prob_uniform, entropy_p, entropy_uniform)`

Calculate Jensen-Shannon divergence between distributions.

**Parameters:**
- `prob_full: NDArray[np.float64]` — Full probability distribution
- `prob_uniform: NDArray[np.float64]` — Uniform probability distribution
- `entropy_p: float` — Entropy of prob_full
- `entropy_uniform: float` — Entropy of prob_uniform

**Returns:**
- `float` — Jensen-Shannon divergence value

---

### `compute_max_divergence(prob_uniform, entropy_uniform)`

Calculate maximum possible Jensen-Shannon divergence.

**Parameters:**
- `prob_uniform: NDArray[np.float64]` — Uniform probability distribution
- `entropy_uniform: float` — Entropy of uniform distribution

**Returns:**
- `float` — Maximum divergence value

---

## Module: `app.features`

Feature extraction functions for audio signal analysis.

### `compute_entropy_complexity(time_series, embedding_dim=3, time_delay=1)`

Compute permutation entropy and statistical complexity using the Bandt-Pompe method.

Uses ordinal patterns to quantify randomness (entropy) and structure (complexity) in time series data. Normalized entropy H ∈ [0,1] and complexity C ∈ [0,1].

**Parameters:**
- `time_series: NDArray[np.float64]` — Input time series signal as 1D array
- `embedding_dim: int` — Embedding dimension for phase space reconstruction. Default is 3
- `time_delay: int` — Time delay between consecutive elements in embedding vectors. Default is 1

**Returns:**
- `tuple[float, float]` — (normalized_entropy, complexity)
  - `normalized_entropy: float` — Shannon entropy normalized by maximum entropy H ∈ [0,1]
  - `complexity: float` — Statistical complexity measure C ∈ [0,1]

**Raises:**
- `ValueError` — If time_series is empty or embedding_dim < 2

**References:**
- Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. Physical Review Letters, 88(17), 174102.

---

### `extract_frequency_features(audio_signal, sample_rate, n_fft=2048)`

Extract frequency domain features from audio signal using Fast Fourier Transform.

Computes spectral features including centroid, bandwidth, rolloff, and energy distribution across frequency bands. Useful for characterizing respiratory sounds.

**Parameters:**
- `audio_signal: NDArray[np.float64]` — Input audio signal as 1D array
- `sample_rate: int` — Sampling rate of the audio in Hz
- `n_fft: int` — FFT window size (number of points). Default is 2048

**Returns:**
- `tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]` — (frequencies, magnitudes, features)
  - `frequencies: NDArray` — Frequency bins in Hz
  - `magnitudes: NDArray` — Magnitude spectrum
  - `features: dict[str, float]` — Dictionary with keys:
    - `'spectral_centroid'` — Center of mass of spectrum (Hz)
    - `'spectral_bandwidth'` — Standard deviation of spectrum (Hz)
    - `'spectral_rolloff'` — Frequency below which 85% of energy exists (Hz)
    - `'low_freq_energy'` — Energy in [0, 500) Hz range
    - `'mid_freq_energy'` — Energy in [500, 1000) Hz range
    - `'high_freq_energy'` — Energy in [1000, ∞) Hz range
    - `'peak_frequency'` — Frequency with maximum magnitude (Hz)

**Raises:**
- `ValueError` — If audio_signal is empty or sample_rate <= 0

---

### `detect_whistles(audio_signal, sample_rate, freq_range=(400, 1600))`

Detect whistle sounds (wheezes) in respiratory audio using spectrogram analysis.

Computes time-frequency representation and measures energy concentration in the typical wheeze frequency range. Wheezes are continuous musical sounds with dominant frequencies typically in 400-1600 Hz range.

**Parameters:**
- `audio_signal: NDArray[np.float64]` — Input audio signal as 1D array
- `sample_rate: int` — Sampling rate of the audio in Hz
- `freq_range: tuple[int, int]` — Tuple (low_freq, high_freq) defining whistle frequency band in Hz. Default is (400, 1600) Hz based on respiratory sound literature

**Returns:**
- `tuple[bool, float]` — (whistle_detected, whistle_strength)
  - `whistle_detected: bool` — True if whistle strength exceeds threshold (0.2)
  - `whistle_strength: float` — Ratio of energy in whistle band to total energy [0,1]

**Raises:**
- `ValueError` — If audio_signal is empty, sample_rate <= 0, or invalid freq_range

**Notes:**
- Detection threshold of 0.2 (20% energy in whistle band) is empirically determined for respiratory sound analysis

---

## Module: `app.data_preprocessing`

Data preprocessing and batch feature extraction for audio signals.

### `extract_features_row(filename, audio_signal, sample_rate, embedding_dim=3, time_delay=1)`

Extract all features from a single audio signal for DataFrame insertion.

Computes frequency features, whistle detection, and entropy-complexity metrics, returning a dictionary suitable for pandas DataFrame row insertion.

**Parameters:**
- `filename: str` — Audio file name (must contain patient ID as prefix: '<id>_...')
- `audio_signal: NDArray[np.float64]` — Input audio signal as 1D array
- `sample_rate: int` — Sampling rate of the audio in Hz
- `embedding_dim: int` — Embedding dimension for entropy-complexity computation. Default is 3
- `time_delay: int` — Time delay for entropy-complexity computation. Default is 1

**Returns:**
- `dict[str, Any]` — Dictionary with keys:
  - `'filename'` — Original filename
  - `'patient_id'` — Extracted patient ID from filename
  - `'low_freq_energy'` — Normalized low frequency energy [0,1]
  - `'mid_freq_energy'` — Normalized mid frequency energy [0,1]
  - `'high_freq_energy'` — Normalized high frequency energy [0,1]
  - `'whistle_strength'` — Whistle detection strength [0,1]
  - `'spectral_centroid'` — Spectral centroid in Hz
  - `'peak_frequency'` — Peak frequency in Hz
  - `'entropy'` — Normalized permutation entropy [0,1]
  - `'complexity'` — Statistical complexity [0,1]

**Raises:**
- `ValueError` — If filename doesn't contain valid patient ID format

---

### `extract_features_batch(signals_dict, embedding_dim=3, time_delay=1)`

Extract features from multiple audio signals in batch.

Processes a dictionary of audio signals and returns a list of feature dictionaries, each representing one audio file's extracted features.

**Parameters:**
- `signals_dict: dict[str, dict[str, Any]]` — Dictionary mapping filenames to signal data:
  - keys: Filenames (str)
  - values: Dict with 'signal' (NDArray) and 'sample_rate' (int)
- `embedding_dim: int` — Embedding dimension for entropy-complexity computation. Default is 3
- `time_delay: int` — Time delay for entropy-complexity computation. Default is 1

**Returns:**
- `list[dict[str, Any]]` — List of feature dictionaries, one per input signal. Each dictionary has the same structure as returned by extract_features_row()

**Example:**
```python
signals = {
    '101_1b1_Al_sc_Meditron.wav': {
        'signal': audio_array,
        'sample_rate': 4000
    }
}
rows = extract_features_batch(signals)
```
