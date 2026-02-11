import numpy as np
from scipy import signal
from scipy.io import wavfile


def load_audio_file(filepath):
    sample_rate, audio_data = wavfile.read(filepath)
    
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    audio_data = audio_data.astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return sample_rate, audio_data


def apply_bandpass_filter(audio_data, sample_rate, lowcut=50, highcut=2000):
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, audio_data)
    
    return filtered_data


def extract_frequency_features(audio_data, sample_rate, n_fft=2048):
    fft_result = np.fft.rfft(audio_data, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1/sample_rate)
    magnitudes = np.abs(fft_result)
    
    feature_vector = {
        'spectral_centroid': np.sum(frequencies * magnitudes) / np.sum(magnitudes),
        'spectral_bandwidth': np.sqrt(np.sum(((frequencies - np.sum(frequencies * magnitudes) / np.sum(magnitudes)) ** 2) * magnitudes) / np.sum(magnitudes)),
        'spectral_rolloff': frequencies[np.where(np.cumsum(magnitudes) >= 0.85 * np.sum(magnitudes))[0][0]],
        'low_freq_energy': np.sum(magnitudes[frequencies < 500]),
        'mid_freq_energy': np.sum(magnitudes[(frequencies >= 500) & (frequencies < 1000)]),
        'high_freq_energy': np.sum(magnitudes[frequencies >= 1000]),
        'peak_frequency': frequencies[np.argmax(magnitudes)]
    }
    
    return frequencies, magnitudes, feature_vector


def detect_whistles(audio_data, sample_rate, whistle_freq_range=(400, 1600)):
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=256)
    
    freq_mask = (f >= whistle_freq_range[0]) & (f <= whistle_freq_range[1])
    whistle_energy = np.sum(Sxx[freq_mask, :])
    total_energy = np.sum(Sxx)
    
    whistle_strength = whistle_energy / total_energy if total_energy > 0 else 0
    whistle_detected = whistle_strength > 0.2
    
    return whistle_detected, whistle_strength


def preprocess_lung_sound(filepath, lowcut=50, highcut=2000):
    sample_rate, audio_data = load_audio_file(filepath)
    
    filtered_data = apply_bandpass_filter(audio_data, sample_rate, lowcut, highcut)
    
    frequencies, magnitudes, freq_features = extract_frequency_features(filtered_data, sample_rate)
    
    whistle_detected, whistle_strength = detect_whistles(filtered_data, sample_rate)
    
    feature_vector = {
        **freq_features,
        'whistle_detected': whistle_detected,
        'whistle_strength': whistle_strength,
        'sample_rate': sample_rate,
        'duration': len(audio_data) / sample_rate
    }
    
    return feature_vector, filtered_data
