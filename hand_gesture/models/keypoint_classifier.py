"""Wrapper for static hand-sign classification with multiple model formats."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from hand_gesture import REPO_ROOT


class KeyPointClassifier:
    """Runs the 42-dim landmark classifier from various model formats.

    Supported formats:
    - .tflite  -> TensorFlow Lite interpreter
    - .h5/.hdf5 -> Keras model (tf.keras.models.load_model)
    - .joblib -> joblib-serialized model (e.g. scikit-learn)
    - .pkl/.pickle -> pickle-serialized model
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        num_threads: int = 1,
    ) -> None:
        # Allow caller to pass a custom path; default to the original TFLite model
        if model_path is None:
            model_path = REPO_ROOT / "models/keypoint/keypoint_classifier.tflite"

        self.model_path = Path(model_path)
        self._backend = self._detect_backend(self.model_path)
        self._num_threads = num_threads

        if self._backend == "tflite":
            # Original TFLite behavior
            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.model_path),
                num_threads=num_threads,
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        elif self._backend == "keras":
            # Keras .h5 / .hdf5 model
            self.model = tf.keras.models.load_model(self.model_path)

        elif self._backend == "joblib":
            # joblib-serialized model (e.g. scikit-learn)
            import joblib

            self.model = joblib.load(self.model_path)

        elif self._backend == "pickle":
            # Generic pickle-serialized model
            import pickle

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model backend: {self._backend}")

    @staticmethod
    def _detect_backend(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".tflite":
            return "tflite"
        if suffix in {".h5", ".hdf5"}:
            return "keras"
        if suffix == ".joblib":
            return "joblib"
        if suffix in {".pkl", ".pickle"}:
            return "pickle"
        raise ValueError(f"Unsupported model format: {suffix}")

    def __call__(self, landmark_list: Sequence[float]) -> int:
        x = np.array([landmark_list], dtype=np.float32)

        if self._backend == "tflite":
            # Original TFLite inference flow
            input_index = self.input_details[0]["index"]
            self.interpreter.set_tensor(input_index, x)
            self.interpreter.invoke()

            output_index = self.output_details[0]["index"]
            result = self.interpreter.get_tensor(output_index)
            return int(np.argmax(np.squeeze(result)))

        # Non-TFLite backends
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("Model is not initialized")

        # Prefer .predict when available (Keras, scikit-learn); otherwise call directly
        if hasattr(model, "predict"):
            y = model.predict(x)
        else:
            y = model(x)

        y = np.array(y)

        # Heuristic: if we get class probabilities/logits, take argmax;
        # if we get class labels, return the first label.
        if y.ndim >= 2:
            # Shape like (1, num_classes) or (num_samples, num_classes)
            logits = y[0] if y.shape[0] == 1 else y
            return int(np.argmax(logits))
        else:
            # Shape like (1,) or scalar
            return int(y[0])
