"""Drawing helpers separated from the main loop."""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2 as cv


def draw_landmarks(image, landmark_point: Sequence[Sequence[int]]):
    if len(landmark_point) == 0:
        return image

    # Thumb
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

    # Index finger
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

    # Middle finger
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

    # Ring finger
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

    # Little finger
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

    # Palm
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in {4, 8, 12, 16, 20} else 5
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(image, brect: Sequence[int], use_brect: bool = True):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text: str, finger_gesture_text: str):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text:
        info_text = f"{info_text}:{hand_sign_text}"
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text:
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_point_history(image, point_history: Sequence[Sequence[int]]):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps: float, mode_id: int, number: int, mode_texts: Dict[int, str]):
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    if mode_id in mode_texts:
        cv.putText(image, f"MODE:{mode_texts[mode_id]}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM:{number}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image
