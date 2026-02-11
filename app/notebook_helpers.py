import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import extract_frequency_features, detect_whistles
from app.entropy_complexity import compute_entropy_complexity


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


def plot_time_frequency_spectra(signals, features_dict, sample_rate):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Frequency Spectrum Analysis of Lung Sounds', fontsize=16, fontweight='bold')
    
    colors_map = {'Healthy': 'green', 'Diseased': 'red', 'COPD': 'orange'}
    
    for idx, (disease_name, signal_data) in enumerate(signals.items()):
        frequencies = features_dict[disease_name]['frequencies']
        magnitudes = features_dict[disease_name]['magnitudes']
        color = colors_map.get(disease_name, 'blue')
        
        axes[idx, 0].plot(signal_data[:1000], linewidth=0.8, color=color)
        axes[idx, 0].set_title(f'{disease_name} - Time Domain')
        axes[idx, 0].set_xlabel('Samples')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        
        axes[idx, 1].plot(frequencies, magnitudes, linewidth=1.2, color=color)
        axes[idx, 1].axvspan(400, 1600, alpha=0.2, color='red', label='Whistle Range')
        axes[idx, 1].axvspan(0, 500, alpha=0.1, color='green', label='Normal Range')
        axes[idx, 1].set_title(f'{disease_name} - Frequency Spectrum')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_xlim(0, 2000)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_energy_distribution(disease_features):
    diseases = list(disease_features.keys())
    colors = ['#2ecc71', '#e74c3c', '#e67e22']
    
    energy_features = ['Low Freq Energy', 'Mid Freq Energy', 'High Freq Energy']
    energy_data = np.array([[disease_features[d][f] for f in energy_features] for d in diseases])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(diseases))
    width = 0.25
    for i, feature in enumerate(energy_features):
        ax.bar(x + i*width, energy_data[:, i], width, label=feature, alpha=0.8)
    
    ax.set_title('Energy Distribution by Frequency Band', fontsize=14, fontweight='bold')
    ax.set_xlabel('Disease Category')
    ax.set_ylabel('Relative Energy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(diseases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_whistle_strength(disease_features):
    diseases = list(disease_features.keys())
    colors = ['#2ecc71', '#e74c3c', '#e67e22']
    whistle_strengths = [disease_features[d]['Whistle Strength'] for d in diseases]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(diseases, whistle_strengths, color=colors, alpha=0.8)
    ax.set_title('Whistle Strength by Disease Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Whistle Strength')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Detection Threshold')
    
    for i, (disease, strength) in enumerate(zip(diseases, whistle_strengths)):
        ax.text(i, strength + 0.03, f'{strength:.3f}', ha='center', fontweight='bold')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_peak_frequencies(disease_features):
    diseases = list(disease_features.keys())
    colors = ['#2ecc71', '#e74c3c', '#e67e22']
    peak_freqs = [disease_features[d]['Peak Frequency'] for d in diseases]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(diseases, peak_freqs, color=colors, alpha=0.8)
    ax.set_title('Peak Frequency by Disease Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')
    ax.axhspan(400, 1600, alpha=0.2, color='red', label='Whistle Range (400-1600 Hz)')
    ax.axhspan(0, 300, alpha=0.2, color='green', label='Normal Range (0-300 Hz)')
    
    for i, (disease, freq) in enumerate(zip(diseases, peak_freqs)):
        ax.text(i, freq + 20, f'{freq:.1f} Hz', ha='center', fontweight='bold')
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_entropy_complexity_plane(disease_features):
    diseases = list(disease_features.keys())
    colors = ['#2ecc71', '#e74c3c', '#e67e22']
    entropies = [disease_features[d]['Entropy'] for d in diseases]
    complexities = [disease_features[d]['Complexity'] for d in diseases]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(entropies, complexities, s=300, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
    
    for i, disease in enumerate(diseases):
        ax.annotate(disease, (entropies[i], complexities[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax.set_title('Entropy-Complexity Plane for Disease Classification', fontsize=14, fontweight='bold')
    ax.set_xlabel('Normalized Entropy (H)', fontsize=12)
    ax.set_ylabel('Statistical Complexity (C)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_whistle_vs_centroid(disease_features):
    diseases = list(disease_features.keys())
    colors = ['#2ecc71', '#e74c3c', '#e67e22']
    whistle_strengths = [disease_features[d]['Whistle Strength'] for d in diseases]
    centroids = [disease_features[d]['Spectral Centroid'] for d in diseases]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(whistle_strengths, centroids, s=300, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
    
    for i, disease in enumerate(diseases):
        ax.annotate(disease, (whistle_strengths[i], centroids[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax.set_title('Whistle Strength vs Spectral Centroid', fontsize=14, fontweight='bold')
    ax.set_xlabel('Whistle Strength', fontsize=12)
    ax.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
