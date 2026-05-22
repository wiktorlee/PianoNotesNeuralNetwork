from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split


class SplitData(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    classes: np.ndarray
    input_mean: float | None
    input_std: float | None


def stratified_subsample_indices(
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    n = len(y)
    if n_samples >= n:
        return np.arange(n)
    idx_sub, _ = train_test_split(
        np.arange(n),
        train_size=n_samples,
        stratify=y,
        random_state=seed,
    )
    return np.asarray(idx_sub, dtype=np.int64)


def resolve_pool_cap(n_pool: int, max_pool_samples: int) -> int:
    if max_pool_samples > 0:
        return min(max_pool_samples, n_pool)
    if n_pool > 20_000:
        return 12_000
    return n_pool


def load_train_val_split(
    data_path: Path,
    *,
    val_fraction: float = 0.1,
    max_pool_samples: int = 0,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    shuffle_index_subset: bool = False,
    standardize_inputs: bool = True,
    seed: int = 42,
) -> SplitData:
    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono datasetu: {data_path}")

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction musi byc w zakresie (0, 1).")

    data = np.load(data_path, allow_pickle=True, mmap_mode="r")
    required_keys = {"X_train", "y_train", "X_val", "y_val", "classes"}
    if not required_keys.issubset(data.files):
        raise ValueError(f"Dataset nie zawiera wymaganych kluczy: {required_keys}")

    X_train_mmap = data["X_train"]
    y_train = np.asarray(data["y_train"])
    X_val_mmap = data["X_val"]
    y_val = np.asarray(data["y_val"])
    classes = np.asarray(data["classes"])

    use_large_val_pool = len(y_val) > max(len(y_train) * 2, 5000)
    if use_large_val_pool:
        pool_y = y_val
        y_source = y_val
        x_source = X_val_mmap
    else:
        pool_y = np.concatenate([y_train, y_val])
        y_source = pool_y
        x_source = np.concatenate(
            [np.asarray(X_train_mmap), np.asarray(X_val_mmap)],
            axis=0,
        )

    pool_cap = resolve_pool_cap(len(pool_y), max_pool_samples)
    if pool_cap < len(pool_y):
        idx_pool = stratified_subsample_indices(pool_y, pool_cap, seed)
        pool_y = pool_y[idx_pool]
    else:
        idx_pool = np.arange(len(pool_y), dtype=np.int64)

    idx_tr_local, idx_va_local = train_test_split(
        np.arange(len(pool_y)),
        test_size=val_fraction,
        random_state=seed,
        stratify=pool_y,
    )

    idx_tr = idx_pool[idx_tr_local]
    idx_va = idx_pool[idx_va_local]

    if shuffle_index_subset:
        rng = np.random.default_rng(seed)
        idx_tr = rng.permutation(idx_tr)
        idx_va = rng.permutation(idx_va)

    if max_train_samples > 0 and max_train_samples < len(idx_tr):
        idx_tr = idx_tr[:max_train_samples]
    if max_val_samples > 0 and max_val_samples < len(idx_va):
        idx_va = idx_va[:max_val_samples]

    if use_large_val_pool:
        X_tr = np.asarray(X_val_mmap[idx_tr], dtype=np.float32)
        y_tr = y_val[idx_tr]
        X_va = np.asarray(X_val_mmap[idx_va], dtype=np.float32)
        y_va = y_val[idx_va]
    else:
        X_tr = np.asarray(x_source[idx_tr], dtype=np.float32)
        y_tr = y_source[idx_tr]
        X_va = np.asarray(x_source[idx_va], dtype=np.float32)
        y_va = y_source[idx_va]

    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_va = np.asarray(y_va, dtype=np.int64)

    mean: float | None = None
    std: float | None = None
    if standardize_inputs:
        mean = float(X_tr.mean())
        std = float(X_tr.std())
        if std < 1e-6:
            std = 1.0
        X_tr = ((X_tr - mean) / std).astype(np.float32)
        X_va = ((X_va - mean) / std).astype(np.float32)

    return SplitData(X_tr, y_tr, X_va, y_va, classes, mean, std)
