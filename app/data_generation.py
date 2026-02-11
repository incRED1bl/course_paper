import numpy as np


def generate_sample_signals(n_points=8000, sample_rate=4000):
    t = np.linspace(0, 2, n_points)
    
    healthy = np.sin(2 * np.pi * 150 * t) + 0.3 * np.random.randn(n_points)
    diseased = np.sin(2 * np.pi * 150 * t) + 0.8 * np.sin(2 * np.pi * 800 * t) + 0.3 * np.random.randn(n_points)
    copd = np.sin(2 * np.pi * 120 * t) + 0.9 * np.sin(2 * np.pi * 600 * t) + 0.4 * np.random.randn(n_points)
    
    return {'Healthy': healthy, 'Diseased': diseased, 'COPD': copd}, sample_rate
