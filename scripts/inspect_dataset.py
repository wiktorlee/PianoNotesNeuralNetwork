"""
Diagnostyka zawartosci datasetu PianoRecordingsSingleNotes.

Zaglada do plikow .pickle i raportuje:
  - typ obiektu
  - ksztalty/dlugosci
  - przykladowe wartosci

Uzycie:
  python scripts/inspect_dataset.py
"""

from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"


def describe(obj, indent: int = 0) -> None:
    """Rekursywnie opisuje strukture obiektu Pythona."""
    pad = "  " * indent

    if isinstance(obj, np.ndarray):
        print(f"{pad}numpy.ndarray  shape={obj.shape}  dtype={obj.dtype}")
        if obj.size > 0:
            print(f"{pad}  min={obj.min():.4f}  max={obj.max():.4f}  mean={obj.mean():.4f}")
        return

    if isinstance(obj, dict):
        print(f"{pad}dict (klucze: {list(obj.keys())[:10]}{'...' if len(obj) > 10 else ''})")
        for k in list(obj.keys())[:5]:
            print(f"{pad}  ['{k}'] ->")
            describe(obj[k], indent + 2)
        return

    if isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}  len={len(obj)}")
        if len(obj) > 0:
            print(f"{pad}  [0] ->")
            describe(obj[0], indent + 2)
        return

    print(f"{pad}{type(obj).__name__}: {repr(obj)[:120]}")


def inspect_pickle(path: Path) -> None:
    print("=" * 80)
    print(f"PLIK: {path.relative_to(PROJECT_ROOT)}")
    print(f"ROZMIAR: {path.stat().st_size / 1e6:.2f} MB")
    print("-" * 80)

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Nie udalo sie wczytac: {e}")
        return

    print(f"TYP GLOWNY: {type(obj).__name__}")
    describe(obj, indent=1)
    print()


def main() -> None:
    if not DATA_RAW_DIR.exists():
        sys.exit(f"[ERROR] Brak folderu {DATA_RAW_DIR}")

    candidates = [
        DATA_RAW_DIR / "Grand" / "DatasetSingleNoteGrand_split_C4.pickle",
        DATA_RAW_DIR / "Grand" / "DatasetSingleNoteGrand_split_A4.pickle",
        DATA_RAW_DIR / "Upright" / "DatasetSingleNote_split_C4.pickle",
    ]

    for path in candidates:
        if path.exists():
            inspect_pickle(path)
        else:
            print(f"[INFO] Pominieto (brak): {path.name}")


if __name__ == "__main__":
    main()
