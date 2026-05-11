"""
Pobiera zbior danych z Kaggle do folderu data_raw/.

Wymagania:
  1. pip install kaggle
  2. Wygenerowany token API: https://www.kaggle.com/settings/account
     -> Create New Token -> kaggle.json
  3. Plik kaggle.json umieszczony w: %USERPROFILE%\.kaggle\kaggle.json
     (Windows)  lub  ~/.kaggle/kaggle.json  (Linux/Mac)

Uzycie:
  python scripts/download_dataset.py
"""

from pathlib import Path
import subprocess
import sys

# UWAGA: ten slug nalezy podmienic na ten Twojego zbioru z Kaggle.
# Slug znajdziesz w URL: https://www.kaggle.com/datasets/<USER>/<SLUG>
KAGGLE_DATASET = "kvpratama/piano-notes"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Pobieram dataset: {KAGGLE_DATASET}")
    print(f"[INFO] Folder docelowy : {DATA_RAW_DIR}")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(DATA_RAW_DIR),
        "--unzip",
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] Nie znaleziono komendy 'kaggle'. Zainstaluj: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle CLI zwrocilo blad (kod {e.returncode}).")
        print("        Sprawdz czy masz plik kaggle.json w %USERPROFILE%\\.kaggle\\")
        sys.exit(1)

    print("[OK] Gotowe. Zawartosc data_raw/:")
    for item in sorted(DATA_RAW_DIR.iterdir()):
        print(f"  - {item.name}")


if __name__ == "__main__":
    main()
