from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_split import load_train_val_split

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_processed" / "dataset.npz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "piano_cnn_best.keras"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "plots"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "models" / "training_history.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation plots for piano CNN.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pool-samples", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Wylacz standaryzacje (domyslnie jak train.py: wlaczona).",
    )
    parser.add_argument(
        "--final-metrics",
        action="store_true",
        help="Metryki koncowe (F1, top-K, macierz %) — domyslnie tylko Mel + rozkład klas.",
    )
    return parser.parse_args()


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top_k = np.argsort(proba, axis=1)[:, -k:]
    hits = np.any(top_k == y_true[:, None], axis=1)
    return float(np.mean(hits))


def save_training_history_plot(history_path: Path, out_dir: Path) -> bool:
    if not history_path.exists():
        print(f"[SKIP] Brak {history_path} — uruchom train.py (bez --no-plots) raz, aby zapisac historie.")
        return False

    hist = np.load(history_path)
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    acc = hist["accuracy"]
    val_acc = hist["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")
    return True


def save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title("Macierz pomylek (liczby)")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums > 0, out=np.zeros_like(cm_norm))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".0%")
    ax.set_title("Macierz pomylek (znormalizowana per klasa)")
    plt.tight_layout()
    path = out_dir / "confusion_matrix_normalized.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def save_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    labels = list(range(len(class_names)))
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wartosc")
    ax.set_title("Metryki per klasa (walidacja)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "per_class_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] {report_path}")


def save_top_k_accuracy_plot(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_dir: Path,
) -> None:
    ks = [1, 2, 3]
    scores = [top_k_accuracy(y_true, proba, k) for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"Top-{k}" for k in ks], scores, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K accuracy (walidacja)")
    ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    path = out_dir / "top_k_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def save_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    n_classes = len(class_names)
    train_counts = np.bincount(y_train.astype(int), minlength=n_classes)
    val_counts = np.bincount(y_val.astype(int), minlength=n_classes)

    x = np.arange(n_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, train_counts, width, label="Train (po split)")
    ax.bar(x + width / 2, val_counts, width, label="Val (po split)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Liczba probek")
    ax.set_title("Rozklad klas w zbiorze")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def save_mel_spectrogram_grid(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    out_dir: Path,
    *,
    seed: int,
) -> None:
    n_classes = len(class_names)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes_flat = axes.ravel()

    for cls in range(n_classes):
        idx_cls = np.where(y == cls)[0]
        ax = axes_flat[cls]
        if len(idx_cls) == 0:
            ax.set_title(f"{class_names[cls]} (brak)")
            ax.axis("off")
            continue
        pick = int(rng.choice(idx_cls))
        spec = X[pick, :, :, 0]
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(class_names[cls])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Przykladowe Mel-spektrogramy (po 1 na klase, val)", y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Amplituda (stand.)")
    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25)
    path = out_dir / "mel_spectrogram_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def save_confidence_histogram(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_dir: Path,
) -> None:
    conf = proba[np.arange(len(y_true)), y_true]
    pred = np.argmax(proba, axis=1)
    correct = pred == y_true
    wrong = ~correct

    fig, ax = plt.subplots(figsize=(8, 4))
    if np.any(correct):
        ax.hist(
            conf[correct],
            bins=15,
            range=(0, 1),
            alpha=0.7,
            label=f"Poprawne ({correct.sum()})",
            color="#2ecc71",
        )
    if np.any(wrong):
        ax.hist(
            proba[wrong].max(axis=1),
            bins=15,
            range=(0, 1),
            alpha=0.7,
            label=f"Bledne ({wrong.sum()})",
            color="#e74c3c",
        )
    ax.set_xlabel("Pewnosc softmax (max dla predykcji)")
    ax.set_ylabel("Liczba probek")
    ax.set_title("Rozklad pewnosci predykcji")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "confidence_histogram.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def save_misclassified_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    out_dir: Path,
    *,
    max_examples: int = 6,
) -> None:
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) == 0:
        print("[INFO] Brak blednych predykcji na val — pomijam misclassified_examples.png")
        note = out_dir / "misclassified_examples.txt"
        note.write_text(
            "Wszystkie probki walidacyjne sklasyfikowane poprawnie.\n",
            encoding="utf-8",
        )
        print(f"[OK] {note}")
        return

    n_show = min(max_examples, len(wrong_idx))
    order = np.argsort(proba[wrong_idx].max(axis=1))[::-1]
    picks = wrong_idx[order[:n_show]]

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, idx in enumerate(picks):
        spec = X[idx, :, :, 0]
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        conf = float(proba[idx, y_pred[idx]])

        axes[0, col].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[0, col].set_title(f"Prawda: {true_name}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].bar(range(len(class_names)), proba[idx], color="#3498db")
        axes[1, col].set_xticks(range(len(class_names)))
        axes[1, col].set_xticklabels(class_names, rotation=90, fontsize=7)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_title(f"Pred: {pred_name} ({conf:.0%})")

    fig.suptitle("Przyklady blednych klasyfikacji", y=1.02)
    plt.tight_layout()
    path = out_dir / "misclassified_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Wczytuje dane: {args.data}")
    split = load_train_val_split(
        args.data,
        val_fraction=args.val_fraction,
        max_pool_samples=args.max_pool_samples,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        standardize_inputs=not args.no_standardize,
        seed=args.seed,
    )
    class_names = [str(c) for c in split.classes]
    y_va = split.y_val

    save_class_distribution(split.y_train, y_va, class_names, out_dir)
    save_mel_spectrogram_grid(split.X_val, y_va, class_names, out_dir, seed=args.seed)

    if args.final_metrics:
        if not args.model.exists():
            raise FileNotFoundError(
                f"Brak modelu: {args.model}\nNajpierw uruchom: python scripts/train.py"
            )
        print(f"[INFO] Wczytuje model: {args.model}")
        model = keras.models.load_model(args.model)
        proba = model.predict(split.X_val, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        val_acc = float(np.mean(y_pred == y_va))
        print(f"[INFO] val_accuracy: {val_acc:.4f}")

        save_training_history_plot(args.history, out_dir)
        save_confusion_matrices(y_va, y_pred, class_names, out_dir)
        save_per_class_metrics(y_va, y_pred, class_names, out_dir)
        save_top_k_accuracy_plot(y_va, proba, out_dir)
        save_confidence_histogram(y_va, proba, out_dir)
        save_misclassified_examples(
            split.X_val, y_va, y_pred, proba, class_names, out_dir
        )
    else:
        print(
            "[INFO] Pomijam metryki koncowe (F1, top-K, ...). "
            "Uzyj --final-metrics lub plot_training_dynamics.py"
        )

    print(f"\n[OK] Wykresy w: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
