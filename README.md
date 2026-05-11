# InstrumentTrainer AI — Faza 1 (Deep Learning)

Klasyfikator wysokości dźwięku pojedynczych nut pianina (12 klas: C, C#, D, D#, E, F, F#, G, G#, A, A#, B) oparty o CNN + Mel-spektrogramy.

Docelowo: silnik aplikacji mobilnej **InstrumentTrainer** (Android/Kotlin), która analizuje dźwięk z mikrofonu w czasie rzeczywistym i daje użytkownikowi feedback o zagranej nucie.

## Stack
- Python 3.11
- TensorFlow / Keras (modelowanie)
- Librosa (przetwarzanie sygnałów)
- NumPy, Matplotlib, scikit-learn
- Kaggle CLI (>=1.8.0) do pobierania datasetu
- Eksport do TensorFlow Lite (`.tflite`) → Android (Kotlin)

## Struktura projektu
```
SieciNeuronowe/
├── data_raw/          # surowe pliki z Kaggle (NIE wrzucane do git)
├── data_processed/    # gotowy zbiór .npz po preprocessingu (NIE wrzucany do git)
├── models/            # zapisane modele .h5, .tflite (NIE wrzucane do git)
├── notebooks/         # eksperymenty w Jupyter
├── scripts/           # skrypty Pythonowe (pipeline)
├── requirements.txt
└── .gitignore
```

## Dataset

Używamy [`riccardosimionato/pianorecordingssinglenotes`](https://www.kaggle.com/datasets/riccardosimionato/pianorecordingssinglenotes) (~4.35 GB).

Struktura po rozpakowaniu:
```
data_raw/
├── PianoRecordingsSingleNotes.wav    # monolityczne nagranie 7 minut @ 48 kHz
├── PianoRecordingsSingleNotes.mid    # MIDI z czasami nut
├── Grand/                            # pianino koncertowe, 24 pickle (12 nut × oktawy 3, 4)
├── Upright/                          # pianino pionowe, 24 pickle
├── Chords/                           # akordy (nieużywane w Fazie 1)
└── Sample-based*/                    # dodatkowe sample
```

Każdy plik `DatasetSingleNote*_split_<NUTA><OKTAWA>.pickle` zawiera dict:
```python
{
    "train": list[7],   # 7 grup dynamiki, każda (6, 200000) sampli
    "val"  : list[7],   # 7 grup dynamiki, każda (1, 200000) sampli
    "max"  : float,     # współczynnik normalizacji
}
```
**~49 wariantów** tej samej nuty per pickle (42 train + 7 val).

### Parametry sygnału (ustalone empirycznie)
- **Sample rate**: 22050 Hz
- **Strojenie**: nieortodoksyjne (A4 ≈ 404 Hz zamiast 440 Hz), ale **wewnętrznie spójne** — sprawdzone testem A3/A4 (stosunek pitchów = 0.5004).

## Setup

### 1. Wirtualne środowisko + zależności
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Token API Kaggle (jednorazowo)
Nowy system tokenów (Kaggle CLI >= 1.8.0):
1. Wejdź na [Kaggle Settings → API Tokens](https://www.kaggle.com/settings/account).
2. Kliknij **Generate New Token** (sekcja „API Tokens (Recommended)").
3. Skopiuj wyświetlony token.
4. Zapisz go (sam ciąg znaków, bez cudzysłowów) do pliku:
   ```
   C:\Users\<user>\.kaggle\access_token
   ```
   Uwaga: bez rozszerzenia `.txt`.

### 3. Pobranie datasetu
```powershell
kaggle datasets download -d riccardosimionato/pianorecordingssinglenotes -p data_raw --unzip
```
Albo przez skrypt:
```powershell
python scripts/download_dataset.py
```

## Pipeline

### Skrypty
- `scripts/download_dataset.py` — pobiera dataset z Kaggle do `data_raw/`.
- `scripts/test_spectrogram.py` — sanity-check, wyświetla pierwszy Mel-spektrogram.
- `scripts/inspect_dataset.py` — diagnostyka struktury pickli.
- `scripts/validate_pickle.py` — wykrywa SR, sprawdza spójność etykietowania (A3/A4 oktawa).
- `scripts/preprocess.py` — przetwarza wszystkie 48 pickli → `data_processed/dataset.npz`.

### Parametry preprocessingu
- **SR** = 22050 Hz
- **Długość okna** = 1.0 s = 22050 sampli
- **Mel-spektrogram**: `n_mels=128`, `n_fft=2048`, `hop_length=512`
- **Output shape**: `(N, 128, 44, 1)` — gotowe dla `Conv2D`

### Mapowanie etykiet (z nazwy pliku → klasa 0–11)
Dataset używa konwencji `d` (diesis) zamiast `#`:
| Plik | Nuta | Klasa |
|------|------|-------|
| `C` | C | 0 |
| `Cd` | C# | 1 |
| `D` | D | 2 |
| `Dd` | D# | 3 |
| `E` | E | 4 |
| `F` | F | 5 |
| `Fd` | F# | 6 |
| `G` | G | 7 |
| `Gd` | G# | 8 |
| `A` | A | 9 |
| `Ad` | A# | 10 |
| `B` | B | 11 |

Oktawa jest redukowana (A3 i A4 → ta sama klasa „A").

## Roadmap

### Faza 1 — Deep Learning (in progress)
- [x] Setup środowiska i pobranie datasetu
- [x] Walidacja struktury pickli i parametrów sygnału (SR, strojenie, spójność etykiet)
- [x] Preprocessing: batch generowanie Mel-spektrogramów (`scripts/preprocess.py`)
- [x] Etykietowanie (parsing nazw plików → klasa 0..11)
- [ ] Model CNN: 3× (Conv2D + MaxPool) + Dense/Softmax (`scripts/train.py`)
- [ ] Trening + wykresy Accuracy/Loss + Confusion Matrix
- [ ] Eksport do `.tflite`

### Faza 2 — integracja Android (planowana)
- [ ] Pre-processing audio z mikrofonu w Kotlinie (resampling, normalizacja stroju)
- [ ] Załadowanie `.tflite` w aplikacji
- [ ] UI z feedbackiem w czasie rzeczywistym

## Status (ostatnia sesja)
- Środowisko gotowe, Kaggle CLI autoryzowany.
- Dataset pobrany, struktura rozpoznana, parametry sygnału ustalone.
- Pipeline preprocessingu napisany — czeka na pierwsze odpalenie i weryfikację wyników.
- **Następny krok**: `python scripts/preprocess.py`, potem `scripts/train.py` (CNN).
