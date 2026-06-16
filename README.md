# Piano note classifier (SN - faza 1)

Projekt na sieci neuronowe. Klasyfikacja pojedynczej nuty pianina na 12 klas (C, C#, D, ..., B) z Mel-spektrogramu. Docelowo mial sluzyc pod aplikacje na Androida (TFLite), na razie jest pipeline w Pythonie.

## Technologie

- Python 3.11
- TensorFlow / Keras
- librosa, numpy, matplotlib, scikit-learn
- Kaggle CLI do pobrania danych

## Struktura

```
SieciNeuronowe/
  data_raw/           surowe dane z Kaggle (nie w git)
  data_processed/     dataset.npz po preprocessingu (nie w git)
  models/             modele i wykresy po treningu (nie w git)
  scripts/
  notebooks/
  requirements.txt
```

## Dataset

Kaggle: riccardosimionato/pianorecordingssinglenotes (ok. 4 GB po rozpakowaniu).

Uzywam folderow Grand/ i Upright/ - po 24 pliki pickle (12 nut, 2 oktawy). W pickle jest dict z kluczami train/val; z kazdego pliku bierzemy tylko prawdziwe nagrania audio z grupy [0], reszte grup odrzucamy (tam byly smieciowe macierze i metadane).

Po preprocessingu:

- X_train: (288, 128, 44, 1)
- X_val: (48, 128, 44, 1)
- razem 336 probek, 12 klas

train.py laczy train+val, robi stratified split (domyslnie 90/10) i wychodzi ok. 302 probek treningowych i 34 walidacyjnych.

Sample rate: 22050 Hz. Strojenie w danych nie jest 440 Hz (A4 ok. 404 Hz), ale jest spojne w calym zbiorze - sprawdzalem validate_pickle.

## Etykiety

Z nazwy pliku (Cd = C#, oktawa odpada):

C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11

## Instalacja

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Token Kaggle (nowy format): wygenerowac na stronie konta, zapisac jako plik:

`C:\Users\<user>\.kaggle\access_token` (bez .txt)

Pobranie danych:

```powershell
kaggle datasets download -d riccardosimionato/pianorecordingssinglenotes -p data_raw --unzip
```

albo `python scripts/download_dataset.py`

## Skrypty

- download_dataset.py - pobranie z Kaggle
- inspect_dataset.py, validate_pickle.py - sprawdzenie pickli
- test_spectrogram.py - podglad mel
- preprocess.py - mel-spektrogramy do data_processed/dataset.npz
- train.py - trening CNN, zapis .keras / .h5 / .tflite
- plot_training_dynamics.py - wykresy z przebiegu uczenia (loss, lr, wagi)
- evaluate_plots.py - mel grid i rozkład klas (domyslnie bez metryk 100%)

Preprocessing: okno 1 s, n_mels=128, n_fft=2048, hop_length=512.

## Model

Proste CNN (nie ResNet itp.):

- 3x Conv2D (32, 64, 128) + MaxPool
- Dense 128 + Dropout 0.3
- softmax 12 klas

Bez BatchNorm - z BN na tym zbiorze wychodzilo slabo. Wejscie standaryzowane (mean/std z train).

## Uruchomienie

```powershell
python scripts/preprocess.py
python scripts/train.py
```

Opcjonalnie po treningu (jesli bylo --no-plots):

```powershell
python scripts/plot_training_dynamics.py
python scripts/evaluate_plots.py
```

Wyniki:

- models/piano_cnn.keras, piano_cnn_best.keras, piano_cnn.tflite
- models/training_history.npz, training_dynamics.npz
- models/plots/ - wykresy (learning_curves, learning_rate, normy wag, macierz pomylek)

train.py z --no-plots pomija generowanie PNG.

## Wyniki (lokalnie)

Na poprawnym dataset.npz: val_accuracy ok. 100% po kilku epokach (EarlyStopping, zwykle 8-10 epok). To na malym val (34 probki) - nie traktowalb tego jako dowodu ze model jest gotowy na mikrofon na zywo.

Losowa accuracy to 1/12 ok. 8.3%.

## Colab

W Colab nie instalowac calego requirements.txt z sztywnym tensorflow - lepiej wbudowane TF + pip install librosa soundfile scikit-learn.

Kolejnosc: preprocess, potem train. Duze zipy i modele nie wrzucac do git (sa w gitignore).

## Podpiecie do aplikacji Android (PianoApp)

Po treningu (`python scripts/train.py`) skopiuj model do repo aplikacji (sciezki wzgledem tego projektu):

```powershell
python scripts/export_android_metadata.py --out ..\PianoApp\app\src\main\assets\model_metadata.json
copy models\piano_cnn.tflite ..\PianoApp\app\src\main\assets\piano_cnn.tflite
```

Bez `piano_cnn.tflite` w `assets/` aplikacja uruchamia sie z mockiem (losowe nuty). Z modelem — TFLite. `model_metadata.json` musi miec `inputMean`/`inputStd` z treningu (eksport skryptem, nie recznie 0/1).

## Co zostalo na pozniej

- sprawozdanie na SN (opis danych, sieci, wykresow, ograniczen)
- ewentualnie wiecej danych / augmentacja

## Status

Faza 1 technicznie zrobiona: preprocess, trening, TFLite, wykresy procesu uczenia. Integracja z PianoApp: skopiuj `.tflite` + `model_metadata.json` (patrz sekcja wyzej). README i kod w repo; dane i modele lokalnie u kazdego kto odtwarza pipeline.
