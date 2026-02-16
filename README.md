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

A signal processing toolkit for extracting features from respiratory audio recordings using advanced time-series analysis methods.

### ✨ Key Features

- 🧮 **Entropy & Complexity** — Bandt-Pompe permutation entropy for time-series analysis
- 📊 **Frequency Analysis** — FFT-based spectral features and energy distribution
- 🎵 **Wheeze Detection** — Automatic detection of whistling sounds and wheezes

---

## 🏗️ Project Structure

```
📦 course_paper
├── 📂 app/
│   ├── computations.py        # Low-level mathematical functions
│   ├── features.py            # Feature computation (entropy, frequency, whistles)
│   └── data_preprocessing.py  # File handling & DataFrame operations
│
└── 📂 colab/
    └── colab_notebook.ipynb   # Google Colab notebook for dataset analysis
```

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

| Function                         | Description                                                            |
| -------------------------------- | ---------------------------------------------------------------------- |
| `compute_entropy_complexity()` | Calculate Bandt-Pompe entropy and statistical complexity               |
| `extract_frequency_features()` | Extract spectral centroid, bandwidth, rolloff, and energy distribution |
| `detect_whistles()`            | Detect wheezing sounds in audio signals                                |

---

## 🛠️ Tech Stack

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

- **Python 3.13+**
- **Poetry** for dependency management

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
