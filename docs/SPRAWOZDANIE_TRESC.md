# Sprawozdanie — Klasyfikacja nut pianina (treść techniczna)

> Dokument do wklejenia w Word/Google Docs/Overleaf. Schemat zgodny ze sprawozdaniem „wykrywanie flag”.
> Pola `[UZUPEŁNIJ]` wypełnij po własnym treningu (`python scripts/train.py`).

---

## Strona tytułowa

| Pole | Wartość |
|------|---------|
| Przedmiot | Podstawy sieci neuronowych |
| Kierunek | Informatyczne Systemy Automatyki |
| Termin zajęć | [UZUPEŁNIJ] |
| Temat | Projekt z sieci głębokich — klasyfikacja nut pianina |
| Skład grupy | Wiktor Lepak 280135, Bartosz Bahonko 280087, Bartosz Wiatrak 280129 |
| Okres | [UZUPEŁNIJ] |
| Repozytorium | https://github.com/wiktorlee/PianoNotesNeuralNetwork |

---

## 1. Wprowadzenie

Projekt dotyczy klasyfikacji pojedynczej nuty pianina na **12 klas chromatycznych** (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) przy użyciu sieci neuronowych głębokich. Wejściem modelu jest **Mel-spektrogram** jednosekundowego fragmentu nagrania audio; wyjściem — prawdopodobieństwo przynależności do jednej z 12 nut (bez rozróżniania oktawy).

Docelowo pipeline ma obsługiwać aplikację mobilną **PianoApp** na Androidzie (inferencja TFLite).

### 1.1 Struktura projektu

- **Przetwarzanie danych** — pobieranie z Kaggle, ekstrakcja audio z pickle, Mel-spektrogramy (`preprocess.py`)
- **Budowa modelu** — CNN dla danych 2D (`train.py`)
- **Trening** — uczenie z callbackami i logowaniem dynamiki (`training_logger.py`)
- **Ewaluacja** — wykresy, macierz pomyłek (`plot_training_dynamics.py`, `evaluate_plots.py`)
- **Eksport mobilny** — TFLite + metadane (`export_android_metadata.py`)

### 1.2 Schemat przepływu

```
Kaggle → download_dataset.py → data_raw/ (pickle)
  → preprocess.py → dataset.npz (Mel-spektrogramy)
  → train.py (stratified split, CNN, callbacks)
  → models/ (.keras, .tflite, wykresy)
  → PianoApp (Android)
```

Plik Graphviz: `docs/schemat_flow.dot` → `dot -Tpdf schemat_flow.dot -o schemat_flow.pdf`

---

## 2. Realizacja projektu

### 2.1 Co zostało zrealizowane

- Automatyczne pobieranie datasetu Kaggle
- Preprocessing audio → Mel-spektrogramy, 12 klas
- Architektura CNN + trening z EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Analiza procesu uczenia (normy wag, learning rate, trajektorie)
- Eksport TFLite i `model_metadata.json` dla Androida
- Notebook Colab (`notebooks/colab_train.ipynb`)

### 2.2 Wykorzystane technologie

| Technologia | Zastosowanie |
|-------------|--------------|
| Python 3.11 | Implementacja pipeline'u |
| TensorFlow/Keras 2.16 | CNN, trening, TFLite |
| librosa | Mel-spektrogramy |
| NumPy | Tablice, dataset.npz |
| scikit-learn | Stratified split, metryki |
| matplotlib | Wykresy |
| Kaggle CLI | Pobieranie danych |

---

## 3. Przetwarzanie danych

### 3.1 Pobieranie danych

- **Źródło:** Kaggle `riccardosimionato/pianorecordingssinglenotes` (~4 GB)
- **Struktura:** foldery `Grand/` i `Upright/`, po 24 pliki pickle (12 nut × 2 oktawy)
- **Format pickle:** dict z kluczami `train`/`val`; z każdej grupy bierzemy tylko prawdziwe nagrania audio (wektory 1D ≥ 10 000 próbek)

### 3.2 Preprocessing

| Parametr | Wartość |
|----------|---------|
| Sample rate | 22 050 Hz |
| Długość fragmentu | 1 s (22 050 próbek) |
| n_mels | 128 |
| n_fft | 2048 |
| hop_length | 512 |
| Kształt wejścia | (128, 44, 1) |
| Skala | power → dB (librosa.power_to_db) |

**Etykiety:** z nazwy pliku (`Cd` → C#); oktawa odrzucona → 12 klas (C=0 … B=11).

**Wynik preprocessingu:**
- X_train: (288, 128, 44, 1), y_train: 288
- X_val: (48, 128, 44, 1), y_val: 48
- Razem: **336 próbek**

**Strojenie:** A4 ≈ 404 Hz (nie 440 Hz), ale spójne w całym zbiorze.

### 3.3 Augmentacja

Nie zaimplementowana w fazie 1. Plan na przyszłość: pitch shift, szum, time stretch.

### 3.4 Podział danych

W `train.py`: połączenie train+val z npz → **stratified split 90/10** (seed=42):
- ~302 próbki treningowe
- ~34 próbki walidacyjne

**Standaryzacja:** X' = (X − μ_train) / σ_train (μ, σ tylko z train).

---

## 4. Budowa i trening modelu

### 4.1 Architektura CNN

```
Input (128, 44, 1)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(12, softmax)
```

- **Bez BatchNorm** — na tym zbiorze pogarszało wyniki
- Ostatnia warstwa w float32 (TFLite / mixed precision)

### 4.2 Proces treningu

| Parametr | Wartość |
|----------|---------|
| Optymalizator | Adam, lr = 0.001 |
| Loss | sparse_categorical_crossentropy |
| Metryka | accuracy |
| Batch size | 64 |
| Max epoki | 30 |
| Baseline (losowy) | ~8.3% (1/12) |
| tf.data | shuffle + prefetch |

**Wyniki:** [UZUPEŁNIJ: liczba epok, najlepsza val_accuracy, epoka najlepsza]

### 4.3 Mechanizmy kontroli

- **ModelCheckpoint** — `piano_cnn_best.keras` (monitor: val_accuracy)
- **EarlyStopping** — patience=5, restore_best_weights
- **ReduceLROnPlateau** — factor=0.5, patience=2, min_lr=1e-6
- **TrainingDynamicsCallback** — lr, normy wag L2, delta norm, próbki wag

**Eksport:** `.keras`, `.h5`, `.tflite`

---

## 5. Ewaluacja modelu

### 5.1 Metryki ogólne

Uruchom: `python scripts/evaluate_plots.py --final-metrics`

| Metryka | Wartość |
|---------|---------|
| Accuracy (val) | [UZUPEŁNIJ] |
| Top-2 Accuracy | [UZUPEŁNIJ] |
| Top-3 Accuracy | [UZUPEŁNIJ] |
| Val loss | [UZUPEŁNIJ] |
| Macro F1 | [UZUPEŁNIJ] |

**Uwaga:** val ma tylko ~34 próbki — wysoka accuracy nie dowodzi działania na mikrofonie na żywo.

### 5.2 Analiza błędów

[UZUPEŁNIJ po treningu — opisz pomyłki lub brak pomyłek]

Potencjalne trudności:
- Sąsiednie półtony (C vs C#)
- Różnice Grand vs Upright
- Mała liczba próbek na klasę

---

## 6. Opracowanie wyników

Wykresy w `models/plots/` (po `python scripts/train.py`):

| Plik | Opis |
|------|------|
| `mel_spectrogram_grid.png` | Po 1 Mel-spektrogram na klasę |
| `class_distribution.png` | Rozkład klas train/val |
| `learning_curves.png` | Loss, accuracy, luki generalizacji |
| `learning_rate.png` | Ewolucja LR |
| `weight_norms_by_layer.png` | Normy L2 wag |
| `weight_delta_norms.png` | Zmiany norm między epokami |
| `weight_samples_trajectory.png` | Trajektorie wybranych wag |
| `confusion_matrix.png` | Macierz pomyłek |
| `per_class_metrics.png` | P/R/F1 per klasa (`--final-metrics`) |
| `top_k_accuracy.png` | Top-K accuracy |
| `confidence_histogram.png` | Rozkład pewności softmax |

### Integracja Android

```powershell
python scripts/export_android_metadata.py --out ..\PianoApp\app\src\main\assets\model_metadata.json
copy models\piano_cnn.tflite ..\PianoApp\app\src\main\assets\piano_cnn.tflite
```

`model_metadata.json` musi zawierać `inputMean`/`inputStd` z treningu.

---

## 7. Wnioski

1. Zbudowano pełny pipeline audio → Mel → CNN → TFLite.
2. Prosta CNN wystarczyła do klasyfikacji 12 nut na małym zbiorze laboratoryjnym.
3. Callbacki (EarlyStopping, ReduceLROnPlateau) skutecznie kontrolują trening.
4. **Ograniczenie:** 336 próbek, val ~34 — wyniki lab nie zastępują testów terenowych.
5. **Dalszy rozwój:** augmentacja, więcej danych, testy w PianoApp z mikrofonem.

---

## Szybka checklista przed oddaniem

- [ ] Uruchomiono `preprocess.py` i `train.py`
- [ ] Wypełniono metryki w sekcjach 4.2, 5.1, 7
- [ ] Wstawiono wykresy z `models/plots/`
- [ ] Wygenerowano schemat z `schemat_flow.dot`
- [ ] Sprawdzono skład grupy i datę na stronie tytułowej
