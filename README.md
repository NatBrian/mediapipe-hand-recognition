# Hand Gesture Recognition (MediaPipe)

This repository contains a modular toolkit for hand-gesture recognition using MediaPipe Hands and TensorFlow Lite. It is structured for real-time inference from a webcam, collecting gesture datasets, and integrating custom models and labels.

## Features
- Real-time 21-landmark detection using MediaPipe Hands.
- Static hand sign recognition using a 42-dimension MLP (`keypoint_classifier.tflite`).
- Dynamic gesture recognition using a 32-dimension MLP over a 16-frame index-finger trajectory (`point_history_classifier.tflite`).
- Data logging functionality via keyboard shortcuts to collect new training data (CSV files).
- Jupyter notebooks for reproducing training and exporting quantized TFLite models.

## Repository layout
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── keypoint.csv                  # static gesture dataset (42 features)
│   ├── point_history.csv             # dynamic gesture dataset (32 features)
│   ├── keypoint_labels.csv           # labels
│   └── point_history_labels.csv      # labels
├── models/                           # Contains pre-trained models
├── notebooks/                        # Contains training notebooks
└── hand_gesture/                     # Source package
      ├── __init__.py
      ├── main.py
      ├── camera.py
      ├── inference.py
      ├── mediapipe_hands.py
      ├── preprocessing.py
      ├── models/
      │   ├── __init__.py
      │   ├── keypoint_classifier.py
      │   └── point_history_classifier.py
      └── utils/
         ├── drawing.py
         └── fps.py
```

## Installation
1. Create a virtual environment (recommended) and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt # Python 3.12.12
   ```

## Usage (Running the demo)

### Normal Mode (Default)
Run the main application module from the repository root:
```bash
python -m hand_gesture.main
```
This mode is for testing and data collection. Gestures are recognized and displayed, but no keyboard shortcuts or mouse control are triggered.

### Application Mode (Shortcuts Enabled)
To enable gesture-controlled shortcuts and mouse control:
```bash
python -m hand_gesture.main --application
```
or using the short form:
```bash
python -m hand_gesture.main -a
```

**Application Mode Features:**
- **Open Hand** → App Switcher (Mac: `Cmd+Tab`, Windows: `Alt+Tab`)
- **OK Sign** → Search (Mac: `Cmd+Space` for Spotlight, Windows: `Win+S`)
- **Peace Sign** → Mission Control/Task View (Mac: `Ctrl+Up`, Windows: `Win+Tab`)
- **Pointer Gesture** → Mouse cursor follows your index finger tip

**Note:** Application mode requires Accessibility permissions on macOS and Windows to control keyboard and mouse. The shortcuts are cross-platform and automatically adapt to your operating system.

### Controls and Data Collection
- `Esc`: Quit the application.
- `n`: Switch to normal inference mode.
- `k`: Switch to keypoint logging mode (`MODE: Logging Key Point`).
- `h`: Switch to point-history logging mode (`MODE: Logging Point History`).
- `0–9`: Assign the pressed class ID to the next logged sample.

When `MODE: Logging Key Point` is active, pressing `0–9` writes a row into `data/keypoint.csv` containing `[class_id, 42 normalized features]`. When `MODE: Logging Point History` is active, it writes `[class_id, 32 normalized features]` describing the last 16 frames of the index-finger tip (captured only when the static classifier predicts class ID `2` – the “Pointer” gesture). Training rows follow the same preprocessing pipeline used at inference time: landmarks are converted to wrist-relative coordinates and normalized by maximum absolute value, and point histories are normalized by frame dimensions before being flattened into feature vectors.

## Training
Two notebooks under `notebooks/` provide a reference training workflow:
1. `keypoint_training.ipynb`: Trains the static hand-sign classifier on `data/keypoint.csv`.
2. `point_history_training.ipynb`: Trains the dynamic fingertip trajectory classifier on `data/point_history.csv`.

Both notebooks use `RANDOM_SEED = 42`, `train_test_split(train_size=0.75)`, and `epochs=1000`, and quantize the exported TFLite models following the same configuration as the runtime pipeline. Adjust `NUM_CLASSES` and the label CSVs under `hand_gesture/data/` whenever you add or remove gesture classes.

## Attribution
This project incorporates and refactors code from [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) by Kazuhito Takahashi (Kazuhito00), licensed under the Apache License 2.0.
