"""
Training CNN modelu do klasyfikacji nut pianina (12 klas).

Domyslnie:
  - wczytuje dataset z data_processed/dataset.npz
  - uzywa duzej puli (oryginalne X_val/y_val) i robi stratified train/val
  - przed limitem probek opcjonalnie tasuje indeksy (--shuffle-index-subset); domyslnie wylaczone
  - domyslnie standaryzuje wejscie X (mean/std z train), co zwykle pomaga po usunieciu BN
  - trenuje CNN (3x Conv2D + ReLU + MaxPool + Dense); bez BatchNorm (stabilniejsze na tym zbiorze)
  - opcjonalnie tf.data (prefetch), mixed precision na GPU, checkpoint najlepszego modelu
  - zapisuje .keras / .h5 / .tflite do models/

Uzycie:
  python scripts/train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_processed" / "dataset.npz"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for piano note classification.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Sciezka do datasetu .npz (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Folder wyjsciowy modelu (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate Adam.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Udzial walidacji z duzej puli X_val/y_val (0-1).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Limit probek treningowych (0 = bez limitu).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Limit probek walidacyjnych (0 = bez limitu).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed dla reproducowalnosci.")
    parser.add_argument(
        "--use-tfdata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uzyj tf.data.Dataset (shuffle + prefetch). Domyslnie wlaczone.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=8192,
        help="Rozmiar bufora shuffle dla tf.data (ograniczany do liczby probek treningowych).",
    )
    parser.add_argument(
        "--shuffle-index-subset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tasuj indeksy przed obcieciem --max-* (domyslnie wylaczone; potrafi popsuc podzbior).",
    )
    parser.add_argument(
        "--standardize-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standaryzacja globalna X: (X-mean)/std liczone na train. Domyslnie wlaczone.",
    )


def build_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    mixed_precision: bool,
) -> keras.Model:
    """CNN z README: 3x (Conv2D + ReLU + MaxPool), Dense, Softmax.

    Bez BatchNorm (na tym zbiorze potrafil utknac na ~losowej dokladnosci).
    """
    _ = mixed_precision  # polityka ustawiana w main(); tu tylko sygnatura dla API.

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    n = len(y)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle and n > 0:
        buf = min(shuffle_buffer, n)
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed) if args.shuffle_index_subset else None

    data_path = args.data
    out_dir = args.out_dir

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono datasetu: {data_path}")

    print(f"[INFO] Wczytuje dataset: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    required_keys = {"X_train", "y_train", "X_val", "y_val", "classes"}
    if not required_keys.issubset(data.files):
        raise ValueError(f"Dataset nie zawiera wymaganych kluczy: {required_keys}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    classes = data["classes"]

    print("[INFO] Oryginalne ksztalty:")
    print(f"  X_train: {X_train.shape} dtype={X_train.dtype}")
    print(f"  y_train: {y_train.shape} dtype={y_train.dtype}")
    print(f"  X_val  : {X_val.shape} dtype={X_val.dtype}")
    print(f"  y_val  : {y_val.shape} dtype={y_val.dtype}")
    print(f"  classes: {classes}")

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction musi byc w zakresie (0, 1).")

    # Korzystamy z duzej puli (oryginalne X_val/y_val), bo X_train ma tylko 720 probek.
    idx_all = np.arange(len(y_val))
    idx_tr, idx_va = train_test_split(
        idx_all,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=y_val,
    )

    print("[INFO] Split z duzej puli (stratified):")
    print(f"  train_idx: {idx_tr.shape[0]}")
    print(f"  val_idx  : {idx_va.shape[0]}")

    if args.shuffle_index_subset:
        assert rng is not None
        idx_tr = rng.permutation(idx_tr)
        idx_va = rng.permutation(idx_va)
        print("[INFO] Zastosowano tasowanie indeksow przed obcieciem (--shuffle-index-subset).")

    if args.max_train_samples > 0 and args.max_train_samples < len(idx_tr):
        idx_tr = idx_tr[: args.max_train_samples]
        print(f"[INFO] Ograniczono train do: {idx_tr.shape[0]} probek")

    if args.max_val_samples > 0 and args.max_val_samples < len(idx_va):
        idx_va = idx_va[: args.max_val_samples]
        print(f"[INFO] Ograniczono val do: {idx_va.shape[0]} probek")

    X_tr = np.asarray(X_val[idx_tr], dtype=np.float32)
    y_tr = np.asarray(y_val[idx_tr], dtype=np.int64)
    X_va = np.asarray(X_val[idx_va], dtype=np.float32)
    y_va = np.asarray(y_val[idx_va], dtype=np.int64)

    if args.standardize_inputs:
        mean = float(X_tr.mean())
        std = float(X_tr.std())
        if std < 1e-6:
            std = 1.0
        X_tr = ((X_tr - mean) / std).astype(np.float32)
        X_va = ((X_va - mean) / std).astype(np.float32)
        print(f"[INFO] Standaryzacja wejscia (train): mean={mean:.6f}, std={std:.6f}")

    print("[INFO] Finalne ksztalty:")
    print(f"  X_tr: {X_tr.shape}  y_tr: {y_tr.shape}")
    print(f"  X_va: {X_va.shape}  y_va: {y_va.shape}")
    bc = np.bincount(y_tr.astype(int), minlength=len(classes))
    print("[INFO] Rozklad klas (train, po obcieciu):", bc.tolist())

    num_classes = len(classes)
    gpus = tf.config.list_physical_devices("GPU")
    print("[INFO] Dostepne GPU:", gpus)

    if args.mixed_precision and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision: mixed_float16")
    else:
        if args.mixed_precision and not gpus:
            print("[WARN] --mixed-precision zignorowane (brak GPU).")

    model = build_model(
        input_shape=X_tr.shape[1:],
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision and bool(gpus),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    best_keras = out_dir / "piano_cnn_best.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_keras),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("[INFO] Start treningu...")
    if args.use_tfdata:
        train_ds = make_tf_dataset(
            X_tr,
            y_tr,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        val_ds = make_tf_dataset(
            X_va,
            y_va,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

    keras_path = out_dir / "piano_cnn.keras"
    h5_path = out_dir / "piano_cnn.h5"
    model.save(keras_path)
    print(f"[OK] Zapisano model (Keras v3): {keras_path}")
    model.save(h5_path)
    print(f"[OK] Zapisano model (legacy HDF5): {h5_path}")
    if best_keras.exists():
        print(f"[OK] Najlepszy checkpoint: {best_keras}")

    # TFLite (mobilny interpreter) oczekuje typowych operatorow w float32.
    # Model w mixed_float16 zostawia w grafie tf.Conv2D w f16 -> blad bez TF Select.
    keras.mixed_precision.set_global_policy("float32")
    export_model = build_model(
        input_shape=X_tr.shape[1:],
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        mixed_precision=False,
    )
    export_model.set_weights(model.get_weights())
    print("[INFO] Eksport TFLite z kopii modelu w float32 (bez mixed precision).")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "piano_cnn.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] Zapisano model TFLite: {tflite_path}")

    best_val_acc = max(history.history.get("val_accuracy", [float("nan")]))
    print(f"[INFO] Najlepsze val_accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
