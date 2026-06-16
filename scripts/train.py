from __future__ import annotations


import argparse
import sys
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_split import load_train_val_split
from training_logger import TrainingDynamicsCallback

PROJECT_ROOT = SCRIPT_DIR.parent
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
    parser.add_argument(
        "--max-pool-samples",
        type=int,
        default=0,
        help=(
            "Limit puli przed splitem (stratified). 0 = auto (12000 gdy X_val>20k). "
            "Wazne dla duzych .npz z Drive — bez tego Colab pada na RAM."
        ),
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
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mixed float16 na GPU (Colab). Domyslnie wylaczone — stabilniejsze na CPU.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Pomin wykresy accuracy/loss i macierz pomylek (np. serwer bez GUI).",
    )
    return parser.parse_args()


def build_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    mixed_precision: bool,
) -> keras.Model:
    _ = mixed_precision

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


def save_training_plots(history: keras.callbacks.History, out_dir: Path) -> None:
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist["loss"], label="train")
    axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, hist["accuracy"], label="train")
    axes[1].plot(epochs, hist["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Wykres treningu: {path}")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title("Confusion Matrix (val)")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Macierz pomylek: {path}")


def save_training_history_npz(history: keras.callbacks.History, out_dir: Path) -> None:
    path = out_dir / "training_history.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in history.history.items()})
    print(f"[OK] Historia treningu: {path}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_path = args.data
    out_dir = args.out_dir

    print(f"[INFO] Wczytuje dataset (mmap): {data_path}")
    split = load_train_val_split(
        data_path,
        val_fraction=args.val_fraction,
        max_pool_samples=args.max_pool_samples,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        shuffle_index_subset=args.shuffle_index_subset,
        standardize_inputs=args.standardize_inputs,
        seed=args.seed,
    )
    X_tr = split.X_train
    y_tr = split.y_train
    X_va = split.X_val
    y_va = split.y_val
    classes = split.classes
    if split.input_mean is not None:
        print(
            f"[INFO] Standaryzacja wejscia (train): mean={split.input_mean:.6f}, "
            f"std={split.input_std:.6f}"
        )

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
    dynamics_cb = TrainingDynamicsCallback(seed=args.seed)

    callbacks = [
        dynamics_cb,
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

    best_val_acc = max(history.history.get("val_accuracy", [float("nan")]))
    print(f"[INFO] Najlepsze val_accuracy: {best_val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_va, verbose=0), axis=1)
    val_acc = float(np.mean(y_pred == y_va))
    print(f"[INFO] val_accuracy (recount): {val_acc:.4f}")

    save_training_history_npz(history, out_dir)
    dynamics_path = out_dir / "training_dynamics.npz"
    dynamics_cb.save_npz(dynamics_path)

    class_names = [str(c) for c in classes]
    if not args.no_plots:
        from plot_training_dynamics import generate_plots

        plots_dir = out_dir / "plots"
        generate_plots(
            out_dir / "training_history.npz",
            dynamics_path,
            plots_dir,
        )
        save_confusion_matrix_plot(y_va, y_pred, class_names, plots_dir)
        print(
            "[INFO] Wykresy danych (Mel, rozkład klas): "
            "python scripts/evaluate_plots.py"
        )

    # Eksport TFLite wykonujemy na koncu, bo na Windows (TF/MLIR) zdarza sie crash.
    # Dzięki temu historia treningu i wykresy sa zapisane nawet gdy konwersja sie nie uda.
    if platform.system().lower().startswith("win"):
        print(
            "[WARN] Pomijam eksport TFLite na Windows (znany crash TF/MLIR). "
            "Model .keras/.h5 oraz wszystkie wykresy sa zapisane."
        )
        return

    keras.mixed_precision.set_global_policy("float32")
    export_model = build_model(
        input_shape=X_tr.shape[1:],
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        mixed_precision=False,
    )
    export_model.set_weights(model.get_weights())
    print("[INFO] Eksport TFLite z kopii modelu w float32 (bez mixed precision).")

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        tflite_model = converter.convert()
        tflite_path = out_dir / "piano_cnn.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[OK] Zapisano model TFLite: {tflite_path}")
    except Exception as exc:
        print(f"[WARN] Nie udalo sie wyeksportowac TFLite: {exc}")
        print("[WARN] Trening, checkpointy i wykresy zostaly zapisane poprawnie.")


if __name__ == "__main__":
    main()
