"""
Preprocessing datasetu PianoRecordingsSingleNotes:

  1. Iteruje po wszystkich pickle'ach z data_raw/Grand/ i data_raw/Upright/.
  2. Z kazdego pickle wyciaga wszystkie nagrania (train + val).
  3. Z kazdego nagrania bierze pierwsza sekunde (22050 sampli).
  4. Liczy Mel-spektrogram (n_mels=128).
  5. Etykietuje nuta z nazwy pliku, redukuje oktawe -> 12 klas (C..B).
  6. Zapisuje data_processed/dataset.npz z X_train, y_train, X_val, y_val.

Wersja oszczedna RAM (Colab): pre-alokacja tablic zamiast list + np.stack.

Uzycie:
  python scripts/preprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import gc
import pickle
import re
import sys
import time

import librosa
import numpy as np


def resolve_project_root() -> Path:
    """Znajdz folder z data_raw/ (Colab: /content lub repo z scripts/)."""
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, script_dir.parent):
        if (root / "data_raw").is_dir():
            return root
    return script_dir.parent


PROJECT_ROOT = resolve_project_root()
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"

SAMPLE_RATE = 22050
DURATION_SEC = 1.0
TARGET_LEN = int(SAMPLE_RATE * DURATION_SEC)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# Drugi wymiar 2D = dlugosc sygnalu w samplach; krotkie wiersze to smieci/metadane w pickle.
MIN_AUDIO_SAMPLES = 10_000

NOTE_TO_CLASS: dict[str, int] = {
    "C": 0, "Cd": 1,
    "D": 2, "Dd": 3,
    "E": 4,
    "F": 5, "Fd": 6,
    "G": 7, "Gd": 8,
    "A": 9, "Ad": 10,
    "B": 11,
}
CLASSES_ORDERED = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FILENAME_RE = re.compile(
    r"DatasetSingleNote(?:Grand)?_split_([A-G]d?)(\d+)\.pickle$",
    re.IGNORECASE,
)


@dataclass
class Stats:
    files_processed: int = 0
    train_samples: int = 0
    val_samples: int = 0
    failures: int = 0


def parse_note_from_filename(path: Path) -> tuple[str, int] | None:
    m = FILENAME_RE.search(path.name)
    if not m:
        return None
    note = m.group(1).capitalize()
    if note[-1:] == "d":
        note = note[0].upper() + "d"
    octave = int(m.group(2))
    return note, octave


def to_mel_spectrogram(audio_1d: np.ndarray) -> np.ndarray:
    if len(audio_1d) < TARGET_LEN:
        audio_1d = np.pad(audio_1d, (0, TARGET_LEN - len(audio_1d)))
    else:
        audio_1d = audio_1d[:TARGET_LEN]

    mel = librosa.feature.melspectrogram(
        y=audio_1d.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def iter_recordings(pickle_path: Path):
    """Tylko prawdziwe nagrania audio — pomija (6250, 6) i inne artefakty w grupach 1-6."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    for split in ("train", "val"):
        if split not in data:
            continue
        for group in data[split]:
            arr = np.asarray(group)
            if arr.ndim == 1:
                if arr.size >= MIN_AUDIO_SAMPLES:
                    yield arr, split
            elif arr.ndim == 2:
                if arr.shape[1] >= MIN_AUDIO_SAMPLES:
                    for row in arr:
                        row = np.asarray(row).reshape(-1)
                        if row.size >= MIN_AUDIO_SAMPLES:
                            yield row, split
                elif arr.shape[0] >= MIN_AUDIO_SAMPLES and arr.shape[1] <= 8:
                    yield arr.reshape(-1), split
            # ndim >= 3 lub macierze typu (6250, 6) — pomijamy


def count_recordings(pickle_path: Path) -> tuple[int, int]:
    n_train, n_val = 0, 0
    for _, split in iter_recordings(pickle_path):
        if split == "train":
            n_train += 1
        else:
            n_val += 1
    return n_train, n_val


def fill_pickle(
    path: Path,
    cls: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    i_train: int,
    i_val: int,
    stats: Stats,
) -> tuple[int, int]:
    for audio, split in iter_recordings(path):
        try:
            mel = to_mel_spectrogram(audio)
        except Exception as e:
            stats.failures += 1
            print(f"  [WARN] {path.name}: blad mel-spect ({e})")
            continue
        if split == "train":
            X_train[i_train, :, :, 0] = mel
            y_train[i_train] = cls
            i_train += 1
            stats.train_samples += 1
        else:
            X_val[i_val, :, :, 0] = mel
            y_val[i_val] = cls
            i_val += 1
            stats.val_samples += 1
    return i_train, i_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-raw",
        type=Path,
        default=None,
        help="Folder z Grand/ i Upright/ (domyslnie: <projekt>/data_raw)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_RAW_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
    if args.data_raw is not None:
        DATA_RAW_DIR = args.data_raw.resolve()
        PROJECT_ROOT = DATA_RAW_DIR.parent
        DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"

    print(f"[INFO] PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"[INFO] DATA_RAW_DIR : {DATA_RAW_DIR}")

    pickle_paths: list[Path] = []
    for subdir in ("Grand", "Upright"):
        pattern = "DatasetSingleNote*_split_*.pickle"
        pickle_paths.extend(sorted((DATA_RAW_DIR / subdir).glob(pattern)))

    if not pickle_paths:
        sys.exit(
            "[ERROR] Brak pickli w data_raw/Grand i data_raw/Upright.\n"
            f"        Szukano w: {DATA_RAW_DIR}\n"
            "        Najpierw pobierz dataset Kaggle do data_raw/ (pickle), "
            "albo wgraj tam rozpakowany zbior.\n"
            "        Sam dataset.npz z Drive NIE wystarczy — preprocess potrzebuje data_raw/."
        )

    print(f"[INFO] Znaleziono {len(pickle_paths)} pickli.")
    print(f"[INFO] SR={SAMPLE_RATE}, duration={DURATION_SEC}s, n_mels={N_MELS}")
    print("[INFO] Licze probki (bez mel-spectrogramow)...")

    total_train = total_val = 0
    for path in pickle_paths:
        n_tr, n_va = count_recordings(path)
        total_train += n_tr
        total_val += n_va

    print(f"[INFO] Oczekiwane probki: train={total_train}, val={total_val}")

    # Wymiar czasowy mel z probki (128, T)
    probe_path = pickle_paths[0]
    probe_mel = None
    for audio, _ in iter_recordings(probe_path):
        probe_mel = to_mel_spectrogram(audio)
        break
    if probe_mel is None:
        sys.exit("[ERROR] Nie udalo sie zbudowac probki mel-spectrogramu.")
    n_frames = probe_mel.shape[1]
    del probe_mel
    gc.collect()

    print(f"[INFO] Alokuje tablice ({n_frames} ramek czasowych)...")
    X_train = np.zeros((total_train, N_MELS, n_frames, 1), dtype=np.float32)
    y_train = np.zeros(total_train, dtype=np.int64)
    X_val = np.zeros((total_val, N_MELS, n_frames, 1), dtype=np.float32)
    y_val = np.zeros(total_val, dtype=np.int64)

    stats = Stats()
    i_train = i_val = 0
    t0 = time.time()

    for i, path in enumerate(pickle_paths, start=1):
        print(f"[{i:>2}/{len(pickle_paths)}] {path.parent.name}/{path.name}")
        parsed = parse_note_from_filename(path)
        if parsed is None:
            print(f"  [WARN] Pomijam (zla nazwa): {path.name}")
            stats.failures += 1
            continue
        note, _ = parsed
        if note not in NOTE_TO_CLASS:
            print(f"  [WARN] Nieznana nuta '{note}' w {path.name}")
            stats.failures += 1
            continue
        cls = NOTE_TO_CLASS[note]
        i_train, i_val = fill_pickle(
            path, cls, X_train, y_train, X_val, y_val, i_train, i_val, stats
        )
        stats.files_processed += 1
        gc.collect()

    elapsed = time.time() - t0

    print()
    print("[INFO] Zapisuje dataset.npz (kompresja, ~1-2 min)...")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "dataset.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        classes=np.array(CLASSES_ORDERED),
    )

    print()
    print("=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    print(f"  Plikow przetworzonych : {stats.files_processed}")
    print(f"  Failures              : {stats.failures}")
    print(f"  Probek treningowych   : {stats.train_samples}")
    print(f"  Probek walidacyjnych  : {stats.val_samples}")
    print(f"  Czas                  : {elapsed:.1f} s")
    print()
    print(f"  X_train shape : {X_train.shape}   dtype={X_train.dtype}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  X_val   shape : {X_val.shape}")
    print(f"  y_val   shape : {y_val.shape}")

    print()
    print("  Rozklad klas (train):")
    for cls_id, name in enumerate(CLASSES_ORDERED):
        count = int(np.sum(y_train == cls_id))
        print(f"    {cls_id:>2}  {name:>3}  : {count}")

    print()
    print(f"[OK] Zapisano: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"     Rozmiar: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
