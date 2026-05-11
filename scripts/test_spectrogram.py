"""
Test sanity-check: wczytuje pierwszy plik .wav z data_raw/
i wyswietla jego Mel-spektrogram.

Cel: upewnic sie, ze Librosa "widzi" Twoje pliki audio.

Uzycie:
  python scripts/test_spectrogram.py
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

SAMPLE_RATE = 22050      # standardowy SR dla librosa
DURATION = 1.0           # 1 sekunda audio
N_MELS = 128             # liczba pasm mel (zgodnie z zalozeniami projektu)
N_FFT = 2048
HOP_LENGTH = 512


def find_first_wav(root: Path) -> Path | None:
    """Zwraca pierwszy znaleziony plik .wav (rekurencyjnie)."""
    for path in sorted(root.rglob("*.wav")):
        return path
    return None


def compute_mel_spectrogram(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def main() -> None:
    if not DATA_RAW_DIR.exists():
        raise SystemExit(f"[ERROR] Brak folderu: {DATA_RAW_DIR}")

    wav_path = find_first_wav(DATA_RAW_DIR)
    if wav_path is None:
        raise SystemExit(
            f"[ERROR] Nie znalazlem zadnego pliku .wav w {DATA_RAW_DIR}.\n"
            "        Pobierz najpierw dataset: python scripts/download_dataset.py"
        )

    print(f"[INFO] Wczytuje: {wav_path.relative_to(PROJECT_ROOT)}")
    mel_db = compute_mel_spectrogram(wav_path)
    print(f"[INFO] Ksztalt spektrogramu: {mel_db.shape}  (n_mels x frames)")

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel-spektrogram: {wav_path.name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
