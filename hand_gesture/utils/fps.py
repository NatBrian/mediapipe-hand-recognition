"""FPS calculator utility."""
from __future__ import annotations

from collections import deque

import cv2 as cv


class CvFpsCalc:
    """FPS calculator utility."""

    def __init__(self, buffer_len: int = 1) -> None:
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._diff_times = deque(maxlen=buffer_len)

    def get(self) -> float:
        current_tick = cv.getTickCount()
        diff_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._diff_times.append(diff_time)
        fps = 1000.0 / (sum(self._diff_times) / len(self._diff_times))
        return round(fps, 2)
