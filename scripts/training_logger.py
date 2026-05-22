from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras


def _optimizer_lr(model: keras.Model) -> float:
    lr = model.optimizer.learning_rate
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    return float(lr)


def _layer_weight_norm(layer: keras.layers.Layer) -> float:
    total = 0.0
    for w in layer.weights:
        arr = w.numpy()
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total)) if total > 0 else 0.0


class TrainingDynamicsCallback(keras.callbacks.Callback):
    def __init__(self, n_weight_samples: int = 8, seed: int = 42) -> None:
        super().__init__()
        self.n_weight_samples = n_weight_samples
        self.rng = np.random.default_rng(seed)
        self._prev_norms: dict[str, float] = {}
        self._sample_specs: list[tuple[str, int, int]] = []
        self.layer_names: list[str] = []
        self.lr: list[float] = []
        self.weight_norm: list[list[float]] = []
        self.weight_delta_norm: list[list[float]] = []
        self.sample_labels: list[str] = []
        self.sample_values: list[list[float]] = []

    def _init_sample_specs(self) -> None:
        candidates: list[tuple[str, keras.layers.Layer, int]] = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            if not weights:
                continue
            flat_len = int(np.prod(weights[0].shape))
            if flat_len < 4:
                continue
            candidates.append((layer.name, layer, flat_len))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[2], reverse=True)
        picked_layers = [candidates[0]]
        if len(candidates) > 1:
            picked_layers.append(candidates[-1])

        per_layer = max(1, self.n_weight_samples // len(picked_layers))
        specs: list[tuple[str, int, int]] = []
        labels: list[str] = []

        for name, layer, flat_len in picked_layers:
            w0 = layer.get_weights()[0].ravel()
            indices = np.unique(
                np.concatenate(
                    [
                        np.array([0, flat_len // 4, flat_len // 2, flat_len - 1], dtype=int),
                        self.rng.integers(0, flat_len, size=per_layer),
                    ]
                )
            )[:per_layer]
            for idx in indices:
                specs.append((name, 0, int(idx)))
                labels.append(f"{name}/W[{idx}]")
                if len(specs) >= self.n_weight_samples:
                    break
            if len(specs) >= self.n_weight_samples:
                break

        self._sample_specs = specs[: self.n_weight_samples]
        self.sample_labels = labels[: len(self._sample_specs)]

    def on_train_begin(self, logs=None) -> None:
        self.layer_names = [layer.name for layer in self.model.layers if layer.weights]
        self._init_sample_specs()
        self._prev_norms = {name: 0.0 for name in self.layer_names}

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.lr.append(_optimizer_lr(self.model))

        norms: list[float] = []
        deltas: list[float] = []
        for layer in self.model.layers:
            if not layer.weights:
                continue
            norm = _layer_weight_norm(layer)
            norms.append(norm)
            prev = self._prev_norms.get(layer.name, norm)
            deltas.append(abs(norm - prev))
            self._prev_norms[layer.name] = norm

        self.weight_norm.append(norms)
        self.weight_delta_norm.append(deltas)

        samples: list[float] = []
        for name, tensor_idx, flat_idx in self._sample_specs:
            layer = self.model.get_layer(name)
            w = layer.get_weights()[tensor_idx].ravel()
            samples.append(float(w[flat_idx]))
        self.sample_values.append(samples)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            lr=np.asarray(self.lr, dtype=np.float64),
            layer_names=np.asarray(self.layer_names, dtype=str),
            weight_norm=np.asarray(self.weight_norm, dtype=np.float64),
            weight_delta_norm=np.asarray(self.weight_delta_norm, dtype=np.float64),
            sample_labels=np.asarray(self.sample_labels, dtype=str),
            sample_values=np.asarray(self.sample_values, dtype=np.float64),
        )
