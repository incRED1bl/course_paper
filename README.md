<div align="center">

# 🫁 Respiratory Sound Analysis

### *Feature extraction from respiratory audio using Bandt-Pompe entropy and statistical complexity*

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Package_Manager-60A5FA?style=for-the-badge&logo=poetry&logoColor=white)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

</div>

---

## 📖 About

A modern signal processing toolkit for extracting features from respiratory audio recordings using advanced time-series analysis methods.

Built with **Python 3.13+** featuring modern type hints and clean code architecture.

### ✨ Key Features

- 🧮 **Entropy & Complexity** — Bandt-Pompe permutation entropy for time-series analysis
- 📊 **Frequency Analysis** — FFT-based spectral features and energy distribution
- 🎵 **Wheeze Detection** — Automatic detection of whistling sounds and wheezes
- 🔍 **Type Safety** — Full type hints using Python 3.9+ syntax (`dict`, `list`, `tuple`)
- 📝 **Clean Code** — Comprehensive docstrings and descriptive function names

---

## 🏗️ Project Structure

```
📦 course_paper
├── 📂 app/
│   ├── __init__.py            # Public API exports
│   ├── computations.py        # Low-level math (entropy, divergence, patterns)
│   ├── features.py            # High-level features (entropy-complexity, FFT, whistles)
│   └── data_preprocessing.py  # Batch processing & DataFrame utilities
│
├── 📂 colab/
│   └── colab_notebook.ipynb   # Google Colab notebook for dataset analysis
│
├── main.py                    # Local entry point
├── pyproject.toml             # Poetry dependencies
└── LICENSE                    # MIT License
```

### Module Overview

- **`computations.py`** — Core mathematical functions with full type hints:
  - `build_embedded_vectors()` — Time-delay embedding
  - `compute_ordinal_patterns()` — Pattern extraction
  - `build_probability_distribution()` — Statistical distributions
  - `compute_shannon_entropy()` — Information theory metrics
  
- **`features.py`** — Feature extraction interface:
  - `compute_entropy_complexity()` — Bandt-Pompe analysis
  - `extract_frequency_features()` — Spectral analysis
  - `detect_whistles()` — Wheeze detection
  
- **`data_preprocessing.py`** — Data pipeline utilities:
  - `extract_features_row()` — Single signal processing
  - `extract_features_batch()` — Batch processing
  - `build_feature_summary()` — Result aggregation

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

```bash
1. Open colab/colab_notebook.ipynb in Google Colab
2. Follow the notebook instructions
```

Perfect for working with the full [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) (3.69GB)

### Option 2: Local Development

```bash
# Install dependencies
poetry install

# Run demo
poetry run python main.py
```

---

## 📚 Core Functions

### Feature Extraction

```python
# Entropy & Complexity Analysis
entropy, complexity = compute_entropy_complexity(
    time_series: NDArray[np.float64],
    embedding_dim: int = 3,
    time_delay: int = 1
) -> tuple[float, float]

# Frequency Domain Features
freqs, mags, features = extract_frequency_features(
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    n_fft: int = 2048
) -> tuple[NDArray, NDArray, dict[str, float]]

# Whistle/Wheeze Detection
detected, strength = detect_whistles(
    audio_signal: NDArray[np.float64],
    sample_rate: int,
    freq_range: tuple[int, int] = (400, 1600)
) -> tuple[bool, float]
```

### Usage Example

```python
from app import compute_entropy_complexity, extract_frequency_features
import numpy as np

# Your audio signal
signal = np.random.randn(1000)
sample_rate = 4000

# Extract features
entropy, complexity = compute_entropy_complexity(signal, embedding_dim=3)
freqs, mags, features = extract_frequency_features(signal, sample_rate)

print(f"Entropy: {entropy:.4f}, Complexity: {complexity:.4f}")
print(f"Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
```

---

## 🛠️ Tech Stack

![NumPy](https://img.shields.io/badge/NumPy-2.4.0-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.17.0-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.0-11557C?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=flat&logo=pandas&logoColor=white)

### Core Technologies

- **Python 3.13+** — Modern type hints (`dict`, `list`, `tuple` syntax)
- **Poetry** — Dependency management and virtual environments
- **NumPy & SciPy** — Numerical computing and signal processing
- **Type Safety** — Full type annotations with `numpy.typing.NDArray`
- **Clean Architecture** — Separation of concerns (computations → features → preprocessing)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
