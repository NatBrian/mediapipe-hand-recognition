"""Wrapper for dynamic gesture classification with multiple model formats."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from hand_gesture import REPO_ROOT


class PointHistoryClassifier:
    """Classifies 16x2 fingertip trajectories with a score threshold.

    Supported model formats:
    - .tflite  -> TensorFlow Lite interpreter
    - .h5/.hdf5 -> Keras model (tf.keras.models.load_model)
    - .joblib -> joblib-serialized model (e.g. scikit-learn)
    - .pkl/.pickle -> pickle-serialized model
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        score_th: float = 0.5,
        invalid_value: int = 0,
        num_threads: int = 1,
    ) -> None:
        # Default to the original TFLite model if no path is provided
        if model_path is None:
            model_path = REPO_ROOT / "models/point_history/point_history_classifier.tflite"

        self.model_path = Path(model_path)
        self._backend = self._detect_backend(self.model_path)
        self._num_threads = num_threads

        self.score_th = score_th
        self.invalid_value = invalid_value

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

    def __call__(self, point_history: Sequence[float]) -> int:
        x = np.array([point_history], dtype=np.float32)

        if self._backend == "tflite":
            # Original TFLite inference flow
            input_index = self.input_details[0]["index"]
            self.interpreter.set_tensor(input_index, x)
            self.interpreter.invoke()

            output_index = self.output_details[0]["index"]
            result = self.interpreter.get_tensor(output_index)
            scores = np.squeeze(result)
            result_index = int(np.argmax(scores))

            if scores[result_index] < self.score_th:
                return self.invalid_value
            return result_index

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
        scores = np.squeeze(y)

        # If we have a score vector (probabilities/logits), apply the same threshold logic
        if scores.ndim == 1 and scores.size > 1:
            result_index = int(np.argmax(scores))
            if scores[result_index] < self.score_th:
                return self.invalid_value
            return result_index

        # Otherwise, treat the model output as a label (no thresholding possible)
        # e.g., y = [class_id] or scalar
        return int(scores if np.isscalar(scores) else scores[0])
