<div align="center">

# 🫁 Respiratory Sound Analysis

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Package_Manager-DE5FE9?style=for-the-badge&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

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
├── pyproject.toml             # uv dependencies
└── LICENSE                    # MIT License
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
uv sync

# Run demo
uv run python main.py
```

---

## 🛠️ Tech Stack

![NumPy](https://img.shields.io/badge/NumPy-2.4.0-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.17.0-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.0-11557C?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=flat&logo=pandas&logoColor=white)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
