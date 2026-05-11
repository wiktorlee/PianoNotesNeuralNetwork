# InstrumentTrainer AI — Faza 1 (Deep Learning)

Klasyfikator wysokości dźwięku pojedynczych nut pianina (12 klas: C, C#, D, D#, E, F, F#, G, G#, A, A#, B) oparty o CNN + Mel-spektrogramy.

## Stack
- Python 3.11
- TensorFlow / Keras
- Librosa, NumPy, Matplotlib
- Eksport do TensorFlow Lite (.tflite) → Android (Kotlin)

## Struktura
```
SieciNeuronowe/
├── data_raw/        # surowe pliki .wav (pobierane z Kaggle)
├── models/          # zapisane modele (.h5, .tflite)
├── notebooks/       # eksperymenty w Jupyter
├── scripts/         # skrypty Pythonowe
└── requirements.txt
```

## Setup

### 1. Wirtualne środowisko + zależności
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Token API Kaggle (jednorazowo)
1. Zaloguj się na [Kaggle](https://www.kaggle.com/settings/account)
2. Sekcja **API** → **Create New Token** → pobierze się `kaggle.json`
3. Przenieś plik do `%USERPROFILE%\.kaggle\kaggle.json`:
```powershell
mkdir $env:USERPROFILE\.kaggle -Force
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

### 3. Pobranie datasetu
Dwie opcje:

**Opcja A — przez nasz skrypt** (najpierw zaktualizuj `KAGGLE_DATASET` w `scripts/download_dataset.py` na slug Twojego zbioru):
```powershell
python scripts/download_dataset.py
```

**Opcja B — bezpośrednio przez Kaggle CLI:**
```powershell
kaggle datasets download -d <USER>/<DATASET_SLUG> -p data_raw --unzip
```
Slug znajdziesz w URL strony datasetu: `https://www.kaggle.com/datasets/<USER>/<DATASET_SLUG>`.

### 4. Test — pierwszy spektrogram
```powershell
python scripts/test_spectrogram.py
```
Powinno wyświetlić okienko Matplotliba z Mel-spektrogramem pierwszego pliku `.wav` z `data_raw/`.

## Roadmap (Faza 1)
- [x] Setup środowiska i pobranie danych
- [ ] Preprocessing: batch generowanie Mel-spektrogramów (n_mels=128, duration=1.0s)
- [ ] Etykietowanie (parsing nazw plików → klasa 0..11)
- [ ] Model CNN: 3× (Conv2D + MaxPool) + Dense/Softmax
- [ ] Trening + wykresy Accuracy/Loss + Confusion Matrix
- [ ] Eksport do `.tflite`
