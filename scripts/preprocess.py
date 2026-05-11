"""
Preprocessing datasetu PianoRecordingsSingleNotes:

  1. Iteruje po wszystkich pickle'ach z data_raw/Grand/ i data_raw/Upright/.
  2. Z kazdego pickle wyciaga wszystkie nagrania (train + val).
  3. Z kazdego nagrania bierze pierwsza sekunde (22050 sampli).
  4. Liczy Mel-spektrogram (n_mels=128).
  5. Etykietuje nuta z nazwy pliku, redukuje oktawe -> 12 klas (C..B).
  6. Zapisuje data_processed/dataset.npz z X_train, y_train, X_val, y_val.

Uzycie:
  python scripts/preprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import sys
import time

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"

SAMPLE_RATE = 22050
DURATION_SEC = 1.0
TARGET_LEN = int(SAMPLE_RATE * DURATION_SEC)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# d (= diesis, czyli #)
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
    """'DatasetSingleNoteGrand_split_Cd4.pickle' -> ('Cd', 4)."""
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
    """Generator: yielduje (audio_1d, split) dla kazdego nagrania w pickle.
    split: 'train' albo 'val'.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    for split in ("train", "val"):
        if split not in data:
            continue
        for group in data[split]:
            arr = np.asarray(group)
            if arr.ndim == 1:
                yield arr, split
            elif arr.ndim == 2:
                for row in arr:
                    yield row, split


def process_pickle(path: Path, stats: Stats) -> tuple[list, list, list, list]:
    parsed = parse_note_from_filename(path)
    if parsed is None:
        print(f"  [WARN] Pomijam (zla nazwa): {path.name}")
        stats.failures += 1
        return [], [], [], []

    note, octave = parsed
    if note not in NOTE_TO_CLASS:
        print(f"  [WARN] Nieznana nuta '{note}' w {path.name}")
        stats.failures += 1
        return [], [], [], []

    cls = NOTE_TO_CLASS[note]

    X_tr, y_tr, X_va, y_va = [], [], [], []
    for audio, split in iter_recordings(path):
        try:
            mel = to_mel_spectrogram(audio)
        except Exception as e:
            stats.failures += 1
            print(f"  [WARN] {path.name}: blad mel-spect ({e})")
            continue
        if split == "train":
            X_tr.append(mel)
            y_tr.append(cls)
            stats.train_samples += 1
        else:
            X_va.append(mel)
            y_va.append(cls)
            stats.val_samples += 1

    stats.files_processed += 1
    return X_tr, y_tr, X_va, y_va


def main() -> None:
    pickle_paths: list[Path] = []
    for subdir in ("Grand", "Upright"):
        pattern = "DatasetSingleNote*_split_*.pickle"
        pickle_paths.extend(sorted((DATA_RAW_DIR / subdir).glob(pattern)))

    if not pickle_paths:
        sys.exit("[ERROR] Nie znalazlem pickli w data_raw/Grand i data_raw/Upright.")

    print(f"[INFO] Znaleziono {len(pickle_paths)} pickli do przetworzenia.")
    print(f"[INFO] SR={SAMPLE_RATE}, duration={DURATION_SEC}s, n_mels={N_MELS}")

    stats = Stats()
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    t0 = time.time()
    for i, path in enumerate(pickle_paths, start=1):
        print(f"[{i:>2}/{len(pickle_paths)}] {path.parent.name}/{path.name}")
        X_tr, y_tr, X_va, y_va = process_pickle(path, stats)
        X_train_all.extend(X_tr)
        y_train_all.extend(y_tr)
        X_val_all.extend(X_va)
        y_val_all.extend(y_va)

    elapsed = time.time() - t0

    X_train = np.stack(X_train_all, axis=0)[..., np.newaxis]
    y_train = np.array(y_train_all, dtype=np.int64)
    X_val = np.stack(X_val_all, axis=0)[..., np.newaxis]
    y_val = np.array(y_val_all, dtype=np.int64)

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

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "dataset.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        classes=np.array(CLASSES_ORDERED),
    )
    print()
    print(f"[OK] Zapisano: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"     Rozmiar: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
