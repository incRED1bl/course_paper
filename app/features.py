import librosa
import numpy as np
import ordpy
from numpy.typing import NDArray

EPSILON_ZERO = 1e-10


class AudioFeatureExtractor:
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        frame_length: int = 2048,
        n_mfcc: int = 13,
        epsilon: float = 1e-6,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        self.epsilon = epsilon

    def _safe_nan_to_num(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.nan_to_num(data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_hjorth_parameters(self, y: NDArray[np.float64]) -> tuple[float, float]:
        if y.size < 3:
            return 0.0, 0.0

        first_derivative = np.diff(y)
        second_derivative = np.diff(first_derivative)

        var_y = np.var(y) + self.epsilon
        var_d1 = np.var(first_derivative) + self.epsilon
        var_d2 = np.var(second_derivative) + self.epsilon

        mobility = np.sqrt(var_d1 / var_y)
        complexity = np.sqrt(var_d2 / var_d1) / (mobility + self.epsilon)
        return float(mobility), float(complexity)

    def extract_golden(self, y: NDArray[np.float64], sample_rate: int) -> dict[str, float]:
        """Extract the recommended compact feature set for respiratory classification."""
        y = self._safe_nan_to_num(y.flatten())

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0].astype(np.float64)

        rmse = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0].astype(np.float64)

        stft_power = np.abs(
            librosa.stft(
                y,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.frame_length,
            )
        ).astype(np.float64) ** 2

        spectral_flatness = librosa.feature.spectral_flatness(S=stft_power)[0].astype(np.float64)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=max(3, self.n_mfcc),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).astype(np.float64)

        hjorth_mobility, _ = self._compute_hjorth_parameters(y)

        features: dict[str, float] = {
            "zcr_variance": float(np.var(zcr)),
            "rmse_coeff_var": float(np.std(rmse) / (np.mean(rmse) + self.epsilon)),
            "spectral_flatness": float(np.mean(spectral_flatness)),
            "hjorth_mobility": float(hjorth_mobility),
        }

        for idx in range(3):
            coeff = mfcc[idx, :]
            features[f"mfcc_{idx + 1}_mean"] = float(np.mean(coeff))
            features[f"mfcc_{idx + 1}_var"] = float(np.var(coeff))

        return {k: float(v) for k, v in features.items()}


def compute_entropy_complexity(
    time_series: NDArray[np.float64],
    embedding_dim: int = 3,
    time_delay: int = 1
) -> tuple[float, float]:
    series = np.nan_to_num(time_series.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if series.size <= (embedding_dim - 1) * time_delay + 1:
        return 0.0, 0.0

    entropy, complexity = ordpy.complexity_entropy(
        series,
        dx=embedding_dim,
        dy=1,
        taux=time_delay,
        tauy=1,
    )
    return float(entropy), float(complexity)


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
