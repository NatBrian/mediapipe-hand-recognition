"""Wrapper for static hand-sign classification with multiple model formats."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from hand_gesture import REPO_ROOT
from hand_gesture.models._lite_runtime import create_tflite_interpreter


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
            # 默认模型（可替换）
            # 说明：仓库中可能包含多种关键点分类模型。
            # - `keypoint_classifier.tflite`        : 原始/默认模型，输入通常为扁平向量 (1, 42)
            # - `keypoint_classifier_1dcnn.tflite` : 1D-CNN 版本，训练时输入为 (21, 2)，推理时需要 (1,21,2)
            # - `keypoint_classifier_mlp.tflite`    : MLP 版本，可能仍接受 (1,42)
            # 要使用 1D-CNN，请把 `model_path` 指向 `models/keypoint/keypoint_classifier_1dcnn.tflite`。
            model_path = REPO_ROOT / "models/keypoint/keypoint_classifier_mlp.tflite"

        self.model_path = Path(model_path)
        self._backend = self._detect_backend(self.model_path)
        self._num_threads = num_threads

        if self._backend == "tflite":
            # Original TFLite behavior
            self.interpreter = create_tflite_interpreter(self.model_path, num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            # 标记当前是否为 1D-CNN 风格的模型（训练时使用形状 (21,2)）
            # 仅当模型文件名为 `keypoint_classifier_1dcnn.tflite` 时，推理前会把 (1,42) 重塑为 (1,21,2)
            self._is_1dcnn = self.model_path.name == "keypoint_classifier_1dcnn.tflite"

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
        # Create a batch axis around the provided (flat) landmark list
        x = np.array([landmark_list], dtype=np.float32)

        if self._backend == "tflite":
            # Original TFLite inference flow
            input_info = self.input_details[0]
            input_index = input_info["index"]

            # 如果是 1D-CNN（训练输入为 (21,2)），则在推理前把 (1,42) 重塑为 (1,21,2)
            if getattr(self, "_is_1dcnn", False):
                declared_shape = tuple(int(s) for s in input_info.get("shape", []))
                if len(declared_shape) >= 3:
                    batch = declared_shape[0] if declared_shape[0] > 0 else 1
                    feature_shape = tuple(declared_shape[1:])
                    expected_feature_size = int(np.prod(feature_shape))

                    # 当输入是扁平 (1, N) 并且 N 等于 feature 尺寸乘积时，重塑为 (batch, *feature_shape)
                    if x.ndim == 2 and x.shape[1] == expected_feature_size:
                        x = x.reshape((batch,) + feature_shape).astype(np.float32)

            # 非 1D-CNN 模型保持原有行为（例如 MLP 接受 (1,42)）
            try:
                self.interpreter.set_tensor(input_index, x)
            except ValueError as e:
                provided_shape = tuple(x.shape)
                raise ValueError(
                    f"Cannot set TFLite tensor: provided shape {provided_shape}, "
                    f"expected {tuple(input_info.get('shape', []))}. Consider returning landmarks as (21,2) or "
                    f"ensure preprocessed input matches the model input shape.") from e

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
