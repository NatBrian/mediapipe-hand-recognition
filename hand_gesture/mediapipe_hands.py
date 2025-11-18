"""MediaPipe Hands wrapper."""
from __future__ import annotations

from typing import Any

import cv2 as cv
import mediapipe as mp


class HandLandmarkDetector:
    """Creates and manages a MediaPipe Hands session."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, image_bgr: "cv.Mat") -> Any:
        """Run landmark detection on a BGR frame (converted internally)."""
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._hands.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        self.close()
