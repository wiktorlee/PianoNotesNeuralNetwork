from pathlib import Path
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

GRAND_A4 = DATA_RAW_DIR / "Grand" / "DatasetSingleNoteGrand_split_A4.pickle"
MONOLITHIC_WAV = DATA_RAW_DIR / "PianoRecordingsSingleNotes.wav"

CANDIDATE_SR = [16000, 22050, 44100, 48000]
A4_FREQ_HZ = 440.0


def load_first_sample(pickle_path: Path) -> np.ndarray:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data["train"][0][0], dtype=np.float32)


def dominant_freq(signal: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)

    spectrum[freqs < 50] = 0
    return float(freqs[np.argmax(spectrum)])


def top_n_freqs(signal: np.ndarray, sr: int, n: int = 5) -> list[tuple[float, float]]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)

    spectrum[freqs < 50] = 0

    n_neighbors = 10
    peak_indices: list[int] = []
    sorted_idx = np.argsort(spectrum)[::-1]
    for idx in sorted_idx:
        if all(abs(idx - p) > n_neighbors for p in peak_indices):
            peak_indices.append(int(idx))
        if len(peak_indices) >= n:
            break
    return [(float(freqs[i]), float(spectrum[i])) for i in peak_indices]


def estimate_pitch_yin(signal: np.ndarray, sr: int) -> float:
    f0 = librosa.yin(
        signal.astype(np.float32),
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    return float(np.median(f0))


def main() -> None:
    print("=" * 70)
    print("KROK 1. Czestotliwosc probkowania monolitycznego .wav")
    print("=" * 70)
    if MONOLITHIC_WAV.exists():
        info = sf.info(str(MONOLITHIC_WAV))
        print(f"  Plik     : {MONOLITHIC_WAV.name}")
        print(f"  SR (Hz)  : {info.samplerate}")
        print(f"  Duration : {info.duration:.2f} s")
        print(f"  Channels : {info.channels}")
    else:
        print("  [WARN] Brak pliku, pomijam.")

    print()
    print("=" * 70)
    print("KROK 2. Pierwsza probka z Grand/A4.pickle")
    print("=" * 70)
    sample = load_first_sample(GRAND_A4)
    print(f"  Plik          : {GRAND_A4.name}")
    print(f"  Ksztalt       : {sample.shape}")
    print(f"  Min/Max/Mean  : {sample.min():.4f} / {sample.max():.4f} / {sample.mean():.4f}")

    print()
    print("=" * 70)
    print("KROK 3. Detekcja SR przez librosa.yin (fundamental, NIE harmoniczne)")
    print("=" * 70)
    print(f"  A4 powinno miec fundament przy ~{A4_FREQ_HZ} Hz.")
    print()
    print(f"  {'SR (Hz)':>10}  {'czas (s)':>10}  {'pitch YIN (Hz)':>16}  ocena")
    print(f"  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*20}")

    best_sr = None
    best_diff = float("inf")
    for sr in CANDIDATE_SR:
        duration = len(sample) / sr
        try:
            f0 = estimate_pitch_yin(sample, sr)
        except Exception as e:
            f0 = float("nan")
            print(f"  {sr:>10}  {duration:>10.2f}  {'YIN error':>16}  {e}")
            continue
        diff = abs(f0 - A4_FREQ_HZ)
        verdict = "<-- A4!" if diff < 5 else ""
        print(f"  {sr:>10}  {duration:>10.2f}  {f0:>16.2f}  {verdict}")
        if diff < best_diff:
            best_diff = diff
            best_sr = sr

    print()
    print(f"  [WYNIK] Najlepszy SR = {best_sr} Hz   (odchyl od 440 Hz: {best_diff:.2f} Hz)")

    print()
    print("-" * 70)
    print(f"  Bonus: top-5 pikow FFT przy SR={best_sr} Hz (harmoniczne A4):")
    print("-" * 70)
    for i, (f, e) in enumerate(top_n_freqs(sample, best_sr, n=5), start=1):
        ratio = f / A4_FREQ_HZ
        print(f"    {i}. {f:8.2f} Hz   energia={e:10.0f}   ~ {ratio:.2f} x 440 Hz")

    print()
    print("=" * 70)
    print("KROK 3b. Spojnosc etykietowania: A3 vs A4 (powinno byc ~0.5x oktawy)")
    print("=" * 70)
    a3_path = DATA_RAW_DIR / "Grand" / "DatasetSingleNoteGrand_split_A3.pickle"
    if a3_path.exists():
        sample_a3 = load_first_sample(a3_path)
        f_a3 = estimate_pitch_yin(sample_a3, best_sr)
        f_a4 = estimate_pitch_yin(sample, best_sr)
        ratio = f_a3 / f_a4
        print(f"  A3 pitch : {f_a3:.2f} Hz")
        print(f"  A4 pitch : {f_a4:.2f} Hz")
        print(f"  stosunek : {ratio:.4f}   (oczekiwany 0.5)")
        if abs(ratio - 0.5) < 0.02:
            print("  [OK] Etykietowanie spojne (A3 jest oktawe nizej niz A4).")
        else:
            print("  [UWAGA] Stosunek odbiega od 0.5 - cos nietypowego.")
    else:
        print(f"  [INFO] Brak pliku A3.pickle, pomijam.")

    print()
    print("=" * 70)
    print("KROK 4. Mel-spektrogram pierwszej 1s przy ustalonym SR")
    print("=" * 70)
    sr = best_sr
    one_sec = sample[: sr * 1]
    if len(one_sec) < sr:
        one_sec = np.pad(one_sec, (0, sr - len(one_sec)))
    mel = librosa.feature.melspectrogram(
        y=one_sec, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"  Mel-spektrogram shape: {mel_db.shape}  (n_mels x frames)")

    output_wav = DATA_RAW_DIR / "_sample_A4_grand.wav"
    sf.write(str(output_wav), one_sec, sr)
    print(f"  Zapisano probke do odsluchu: {output_wav.name}")

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Grand A4 -- pierwsza 1s @ SR={sr} Hz")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
