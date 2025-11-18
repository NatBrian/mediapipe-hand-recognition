"""Logic helpers for gesture inference and dataset logging."""
from __future__ import annotations

import csv
from collections import Counter, deque
from enum import IntEnum
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

from hand_gesture import REPO_ROOT, DEFAULT_HISTORY_LENGTH, POINTER_GESTURE_ID
# DEFAULT_HISTORY_LENGTH: Controls the number of historical frames (16) used for tracking finger movements and stabilizing gesture recognition.
# POINTER_GESTURE_ID: Identifies the "pointer" gesture (ID `2`), which triggers the tracking of the index finger's movement for drawing trajectories or recognizing patterns.
from hand_gesture.preprocessing import pre_process_point_history


class LoggingMode(IntEnum):
    """Keyboard-controlled application mode."""

    NORMAL = 0
    KEYPOINT = 1
    POINT_HISTORY = 2


class GestureHistory:
    """Maintains fingertip and classifier histories."""

    def __init__(
        self,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        pointer_gesture_id: int = POINTER_GESTURE_ID,
        tracked_landmark_index: int = 8,
    ) -> None:
        self.history_length = history_length
        self.pointer_gesture_id = pointer_gesture_id
        self.tracked_landmark_index = tracked_landmark_index
        self.point_history: Deque[Sequence[int]] = deque(maxlen=history_length)
        self.finger_gesture_history: Deque[int] = deque(maxlen=history_length)

    def preprocess_point_history(self, image) -> List[float]:
        return pre_process_point_history(image, self.point_history)

    def update_point_history(self, landmark_list: Sequence[Sequence[int]], hand_sign_id: int) -> None:
        if hand_sign_id == self.pointer_gesture_id and len(landmark_list) > self.tracked_landmark_index:
            self.point_history.append(landmark_list[self.tracked_landmark_index])
        else:
            self.point_history.append([0, 0])

    def mark_no_hand(self) -> None:
        self.point_history.append([0, 0])

    def classify_finger_gesture(self, classifier, preprocessed_history: Sequence[float]) -> Tuple[int, int]:
        finger_gesture_id = 0
        if len(preprocessed_history) == (self.history_length * 2):
            finger_gesture_id = classifier(preprocessed_history)

        self.finger_gesture_history.append(finger_gesture_id)
        most_common = Counter(self.finger_gesture_history).most_common()
        stabilized_id = most_common[0][0] if most_common else finger_gesture_id
        return finger_gesture_id, stabilized_id


def select_mode(key: int, mode: LoggingMode) -> Tuple[int, LoggingMode]:
    number = -1
    if 48 <= key <= 57:  # 0-9
        number = key - 48
    if key == ord('n'):
        mode = LoggingMode.NORMAL
    if key == ord('k'):
        mode = LoggingMode.KEYPOINT
    if key == ord('h'):
        mode = LoggingMode.POINT_HISTORY
    return number, mode


def log_sample(
    number: int,
    mode: LoggingMode,
    landmark_list: Sequence[float],
    point_history_list: Sequence[float],
    keypoint_csv: Optional[Path] = None,
    point_history_csv: Optional[Path] = None,
) -> None:
    if not (0 <= number <= 9):
        return

    keypoint_csv = keypoint_csv or REPO_ROOT / 'data/keypoint.csv'
    point_history_csv = point_history_csv or REPO_ROOT / 'data/point_history.csv'

    if mode == LoggingMode.KEYPOINT:
        with keypoint_csv.open('a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == LoggingMode.POINT_HISTORY:
        with point_history_csv.open('a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])


def load_labels(csv_path: Path) -> List[str]:
    with csv_path.open(encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]
