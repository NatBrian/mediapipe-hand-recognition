"""Pre-processing helpers"""
from __future__ import annotations

import copy
import itertools
from typing import Deque, List, Sequence

import cv2 as cv
import numpy as np


def calc_bounding_rect(image: "cv.Mat", landmarks) -> List[int]:
    """Compute the axis-aligned bounding box around detected landmarks."""
    image_height, image_width = image.shape[0], image.shape[1]
    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image: "cv.Mat", landmarks) -> List[List[int]]:
    """Convert MediaPipe normalized landmarks to absolute pixel coordinates."""
    image_height, image_width = image.shape[0], image.shape[1]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmarks(landmark_list: Sequence[Sequence[int]]) -> List[float]:
    """Match the relative, normalized preprocessing of the original script."""
    temp_landmarks = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmarks):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmarks[index][0] = temp_landmarks[index][0] - base_x
        temp_landmarks[index][1] = temp_landmarks[index][1] - base_y

    temp_landmarks = list(itertools.chain.from_iterable(temp_landmarks))
    max_value = max(list(map(abs, temp_landmarks))) if temp_landmarks else 1.0

    def _normalize(value: float) -> float:
        return value / max_value if max_value != 0 else 0.0

    return list(map(_normalize, temp_landmarks))


def pre_process_point_history(image: "cv.Mat", point_history: Deque[Sequence[int]]) -> List[float]:
    """Normalize fingertip trajectories relative to the origin sample and image size."""
    image_height, image_width = image.shape[0], image.shape[1]
    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    return list(itertools.chain.from_iterable(temp_point_history))
