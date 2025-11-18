"""Camera helpers for webcam capture."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2 as cv
import cv2.typing


class Camera:
    """Thin wrapper around ``cv.VideoCapture`` with mirroring support."""

    def __init__(
        self,
        device: int = 0,
        width: int = 960,
        height: int = 540,
        mirror: bool = True,
    ) -> None:
        self._device = device
        self._mirror = mirror
        self._capture = cv.VideoCapture(device)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[bool, Optional[cv.typing.MatLike]]:
        """Return the next frame, optionally mirrored like the original demo."""
        success, frame = self._capture.read()
        if not success:
            return success, None
        if self._mirror:
            frame = cv.flip(frame, 1)
        return success, frame

    def release(self) -> None:
        """Release camera resources."""
        if self._capture:
            self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        self.release()
