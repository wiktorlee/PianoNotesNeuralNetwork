"""Eksport model_metadata.json dla aplikacji Android InstrumentTrainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_split import load_train_val_split

CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_processed" / "dataset.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Sciezka wyjsciowa JSON (domyslnie stdout)",
    )
    args = parser.parse_args()

    split = load_train_val_split(args.data)
    if split.input_mean is None or split.input_std is None:
        raise SystemExit("Brak standaryzacji — uruchom train z --standardize-inputs")

    metadata = {
        "modelFile": "piano_cnn.tflite",
        "sampleRate": 22050,
        "durationSec": 1.0,
        "nMels": 128,
        "nFft": 2048,
        "hopLength": 512,
        "inputMean": split.input_mean,
        "inputStd": split.input_std,
        "classLabels": CLASS_LABELS,
    }

    payload = json.dumps(metadata, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"[OK] Zapisano: {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
