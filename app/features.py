import librosa
import numpy as np
import ordpy
from numpy.typing import NDArray
from scipy import signal
from scipy.stats import kurtosis, skew

from .computations import EPSILON_ZERO


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

    def _safe_stats(self, values: NDArray[np.float64], prefix: str) -> dict[str, float]:
        if values.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_var": 0.0,
                f"{prefix}_skew": 0.0,
                f"{prefix}_kurtosis": 0.0,
            }

        values = values.astype(np.float64)
        mean_value = float(np.mean(values))
        var_value = float(np.var(values))
        skew_value = float(skew(values, bias=False, nan_policy="omit"))
        kurtosis_value = float(kurtosis(values, fisher=True, bias=False, nan_policy="omit"))

        skew_value = float(np.nan_to_num(skew_value, nan=0.0, posinf=0.0, neginf=0.0))
        kurtosis_value = float(np.nan_to_num(kurtosis_value, nan=0.0, posinf=0.0, neginf=0.0))

        return {
            f"{prefix}_mean": mean_value,
            f"{prefix}_var": var_value,
            f"{prefix}_skew": skew_value,
            f"{prefix}_kurtosis": kurtosis_value,
        }

    def _safe_nan_to_num(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.nan_to_num(data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_tkeo_mean(self, y: NDArray[np.float64]) -> float:
        if y.size < 3:
            return 0.0
        tkeo = (y[1:-1] ** 2) - (y[:-2] * y[2:])
        return float(np.mean(tkeo))

    def _compute_spectral_entropy(self, spectrum: NDArray[np.float64]) -> float:
        p = spectrum / (np.sum(spectrum) + self.epsilon)
        p = self._safe_nan_to_num(p)
        entropy = -np.sum(p * np.log(p + self.epsilon))
        return float(entropy / (np.log(len(p) + self.epsilon) + self.epsilon))

    def _compute_spectral_slope(
        self,
        freqs: NDArray[np.float64],
        power_spectrum: NDArray[np.float64],
    ) -> float:
        x = freqs.astype(np.float64)
        y = power_spectrum.astype(np.float64)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + self.epsilon
        return float(numerator / denominator)

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

    def _compute_hnr(self, y: NDArray[np.float64], sample_rate: int) -> float:
        y_centered = y - np.mean(y)
        autocorr = np.correlate(y_centered, y_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]

        if autocorr.size == 0:
            return 0.0

        r0 = autocorr[0] + self.epsilon
        min_lag = max(1, sample_rate // 500)
        max_lag = min(len(autocorr) - 1, sample_rate // 50)
        if max_lag <= min_lag:
            return 0.0

        harmonic_peak = np.max(autocorr[min_lag:max_lag])
        noise_component = max(r0 - harmonic_peak, self.epsilon)
        harmonic_component = max(harmonic_peak, self.epsilon)
        return float(10.0 * np.log10(harmonic_component / noise_component))

    def extract(self, y: NDArray[np.float64], sample_rate: int) -> dict[str, float]:
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

        stft_mag = np.abs(
            librosa.stft(
                y,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.frame_length,
            )
        ).astype(np.float64)
        stft_power = (stft_mag ** 2).astype(np.float64)

        spectral_flatness = librosa.feature.spectral_flatness(S=stft_power)[0].astype(np.float64)
        spectral_crest = np.max(stft_mag, axis=0) / (np.mean(stft_mag, axis=0) + self.epsilon)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).astype(np.float64)
        delta_mfcc = librosa.feature.delta(mfcc).astype(np.float64)

        mean_power_spectrum = np.mean(stft_power, axis=1)
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft).astype(np.float64)

        features: dict[str, float] = {}

        # Unified aggregated stats for base frame-wise features.
        features.update(self._safe_stats(zcr, "zcr"))
        features.update(self._safe_stats(rmse, "rmse"))
        features.update(self._safe_stats(spectral_flatness, "spectral_flatness"))

        # Required explicit formulas.
        features["zcr_variance"] = float(np.var(zcr))
        features["rmse_coeff_var"] = float(np.std(rmse) / (np.mean(rmse) + self.epsilon))
        features["tkeo_mean"] = self._compute_tkeo_mean(y)
        features["spectral_crest_factor_mean"] = float(np.mean(spectral_crest))
        features["spectral_crest_factor_var"] = float(np.var(spectral_crest))
        features["spectral_entropy"] = self._compute_spectral_entropy(mean_power_spectrum)
        features["spectral_slope"] = self._compute_spectral_slope(frequencies, mean_power_spectrum)

        # MFCC + Delta-MFCC statistics per coefficient.
        for idx in range(self.n_mfcc):
            coeff = mfcc[idx, :]
            features.update(self._safe_stats(coeff, f"mfcc_{idx + 1}"))

            delta_coeff = delta_mfcc[idx, :]
            features.update(self._safe_stats(delta_coeff, f"delta_mfcc_{idx + 1}"))

        hjorth_mobility, hjorth_complexity = self._compute_hjorth_parameters(y)
        features["hjorth_mobility"] = hjorth_mobility
        features["hjorth_complexity"] = hjorth_complexity
        features["hnr"] = self._compute_hnr(y, sample_rate)

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
