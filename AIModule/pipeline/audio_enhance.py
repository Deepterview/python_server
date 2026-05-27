import numpy as np
import librosa
import soundfile as sf


def enhance_audio(audio_path: str, output_path: str) -> str:
    y, sr = sf.read(audio_path)
    y = y.astype(np.float32)
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    noise_sample = y[:int(sr * 0.5)]
    noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1, keepdims=True)

    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)

    noise_profile_full = np.repeat(noise_profile, magnitude.shape[1], axis=1)
    cleaned = np.maximum(magnitude - noise_profile_full * 0.5, 0)
    D_cleaned = cleaned * np.exp(1j * phase)
    y_cleaned = librosa.istft(D_cleaned)

    target_rms = 0.05
    current_rms = np.sqrt(np.mean(y_cleaned ** 2))
    if current_rms > 0:
        y_cleaned = y_cleaned * (target_rms / current_rms)

    y_cleaned = np.clip(y_cleaned, -1.0, 1.0)
    sf.write(output_path, y_cleaned, sr)
    return output_path