"""
Respiratory Sound Analysis - Local Demo

Example usage of the feature extraction API.
"""
import numpy as np

from app import (
    compute_entropy_complexity,
    extract_frequency_features,
    detect_whistles,
)


def main() -> None:
    """Run a simple demo of feature extraction."""
    # Generate sample signal (in real use, load from audio file)
    sample_rate = 4000
    duration = 1.0  # seconds
    signal = np.random.randn(int(sample_rate * duration))
    
    print("🫁 Respiratory Sound Analysis Demo\n")
    print(f"Signal length: {len(signal)} samples")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} sec\n")
    
    # Extract entropy and complexity
    print("📊 Computing entropy-complexity features...")
    entropy, complexity = compute_entropy_complexity(
        signal, 
        embedding_dim=3, 
        time_delay=1
    )
    print(f"  Normalized Entropy: {entropy:.4f}")
    print(f"  Statistical Complexity: {complexity:.4f}\n")
    
    # Extract frequency features
    print("🎵 Extracting frequency domain features...")
    frequencies, magnitudes, freq_features = extract_frequency_features(
        signal, 
        sample_rate
    )
    print(f"  Spectral Centroid: {freq_features['spectral_centroid']:.2f} Hz")
    print(f"  Spectral Bandwidth: {freq_features['spectral_bandwidth']:.2f} Hz")
    print(f"  Peak Frequency: {freq_features['peak_frequency']:.2f} Hz\n")
    
    # Detect whistles/wheezes
    print("🔍 Detecting whistles/wheezes...")
    whistle_detected, whistle_strength = detect_whistles(signal, sample_rate)
    print(f"  Whistle Detected: {whistle_detected}")
    print(f"  Whistle Strength: {whistle_strength:.4f}\n")
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()