from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = PROJECT_ROOT / "models" / "training_history.npz"
DEFAULT_DYNAMICS = PROJECT_ROOT / "models" / "training_dynamics.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training process dynamics.")
    p.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def _epochs(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def plot_learning_curves(hist: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    acc = hist["accuracy"]
    val_acc = hist["val_accuracy"]
    ep = _epochs(len(loss))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ep, loss, "o-", label="train", markersize=4)
    axes[0, 0].plot(ep, val_loss, "o-", label="val", markersize=4)
    axes[0, 0].set_title("Strata (loss)")
    axes[0, 0].set_xlabel("Epoka")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, acc, "o-", label="train", markersize=4)
    axes[0, 1].plot(ep, val_acc, "o-", label="val", markersize=4)
    axes[0, 1].set_title("Dokladnosc (accuracy)")
    axes[0, 1].set_xlabel("Epoka")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    gap_loss = np.asarray(val_loss) - np.asarray(loss)
    axes[1, 0].plot(ep, gap_loss, "o-", color="#e67e22", markersize=4)
    axes[1, 0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1, 0].set_title("Luka generalizacji: val_loss - train_loss")
    axes[1, 0].set_xlabel("Epoka")
    axes[1, 0].grid(alpha=0.3)

    gap_acc = np.asarray(val_acc) - np.asarray(acc)
    axes[1, 1].plot(ep, gap_acc, "o-", color="#9b59b6", markersize=4)
    axes[1, 1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1, 1].set_title("Luka: val_accuracy - train_accuracy")
    axes[1, 1].set_xlabel("Epoka")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "learning_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def plot_learning_rate(dyn: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    lr = dyn["lr"]
    ep = _epochs(len(lr))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, lr, "o-", color="#2980b9", markersize=5)
    ax.set_yscale("log")
    ax.set_title("Wspolczynnik uczenia (Adam + ReduceLROnPlateau)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.3, which="both")

    changes = np.where(np.diff(lr) != 0)[0] + 1
    for e in changes:
        ax.axvline(e, color="#e74c3c", ls=":", alpha=0.6)

    plt.tight_layout()
    path = out_dir / "learning_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def plot_weight_norms(dyn: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    names = [str(n) for n in dyn["layer_names"]]
    norms = dyn["weight_norm"]
    ep = _epochs(norms.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(names):
        ax.plot(ep, norms[:, j], "o-", label=name, markersize=3, linewidth=1.2)
    ax.set_title("Norma L2 wag per warstwa (||W||)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("||W||_2")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "weight_norms_by_layer.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def plot_weight_delta_norms(dyn: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    names = [str(n) for n in dyn["layer_names"]]
    deltas = dyn["weight_delta_norm"]
    ep = _epochs(deltas.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(names):
        ax.plot(ep, deltas[:, j], "o-", label=name, markersize=3, linewidth=1.2)
    ax.set_title("Zmiana normy wag miedzy epokami (| ||W_t|| - ||W_{t-1}|| |)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Delta ||W||")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "weight_delta_norms.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def plot_weight_samples(dyn: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    labels = [str(x) for x in dyn["sample_labels"]]
    values = dyn["sample_values"]
    if values.size == 0:
        print("[SKIP] Brak probek wag do wykresu trajektorii.")
        return

    ep = _epochs(values.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, label in enumerate(labels):
        ax.plot(ep, values[:, j], "o-", label=label, markersize=3, linewidth=1.2)
    ax.set_title("Trajektorie wybranych wag (probki z warstw Conv/Dense)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Wartosc wagi")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "weight_samples_trajectory.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] {path}")


def generate_plots(
    history_path: Path,
    dynamics_path: Path,
    out_dir: Path,
) -> None:
    if not history_path.exists():
        raise FileNotFoundError(
            f"Brak {history_path}\nUruchom: python scripts/train.py"
        )
    if not dynamics_path.exists():
        raise FileNotFoundError(
            f"Brak {dynamics_path}\nUruchom: python scripts/train.py"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    hist = np.load(history_path)
    dyn = np.load(dynamics_path, allow_pickle=True)

    plot_learning_curves(hist, out_dir)
    plot_learning_rate(dyn, out_dir)
    plot_weight_norms(dyn, out_dir)
    plot_weight_delta_norms(dyn, out_dir)
    plot_weight_samples(dyn, out_dir)
    hist.close()
    dyn.close()
    print(f"\n[OK] Wykresy procesu uczenia: {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    generate_plots(args.history, args.dynamics, args.out_dir)


if __name__ == "__main__":
    main()
