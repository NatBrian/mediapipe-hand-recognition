from __future__ import annotations

import argparse
import copy

import cv2 as cv

from hand_gesture import PACKAGE_ROOT, REPO_ROOT
from hand_gesture.camera import Camera
from hand_gesture.inference import GestureHistory, LoggingMode, load_labels, log_sample, select_mode
from hand_gesture.mediapipe_hands import HandLandmarkDetector
from hand_gesture.models import KeyPointClassifier, PointHistoryClassifier
from hand_gesture.preprocessing import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmarks,
)
from hand_gesture.utils.drawing import (
    draw_bounding_rect,
    draw_info,
    draw_info_text,
    draw_landmarks,
    draw_point_history,
)
from hand_gesture.utils.fps import CvFpsCalc

DEVICE = 0
CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
KEYPOINT_MODEL_PATH = REPO_ROOT / 'models/keypoint/keypoint_classifier.tflite'
POINT_HISTORY_MODEL_PATH = REPO_ROOT / 'models/point_history/point_history_classifier.tflite'
KEYPOINT_LABEL_PATH = REPO_ROOT / 'data/keypoint_labels.csv'
POINT_HISTORY_LABEL_PATH = REPO_ROOT / 'data/point_history_labels.csv'
KEYPOINT_CSV_PATH = REPO_ROOT / 'data/keypoint.csv'
POINT_HISTORY_CSV_PATH = REPO_ROOT / 'data/point_history.csv'

MODE_TEXT = {
    LoggingMode.KEYPOINT: "Logging Key Point",
    LoggingMode.POINT_HISTORY: "Logging Point History",
}

def main() -> None:
    keypoint_labels = load_labels(KEYPOINT_LABEL_PATH)
    point_history_labels = load_labels(POINT_HISTORY_LABEL_PATH)

    keypoint_classifier = KeyPointClassifier(model_path=KEYPOINT_MODEL_PATH)
    point_history_classifier = PointHistoryClassifier(model_path=POINT_HISTORY_MODEL_PATH)

    fps_calc = CvFpsCalc(buffer_len=10)
    gesture_history = GestureHistory()

    with Camera(DEVICE, CAPTURE_WIDTH, CAPTURE_HEIGHT) as camera, HandLandmarkDetector(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as detector:
        mode = LoggingMode.NORMAL

        while True:
            fps = fps_calc.get()

            key = cv.waitKey(10)
            if key == 27:
                break
            number, mode = select_mode(key, mode)

            success, frame = camera.read()
            if not success or frame is None:
                break
            debug_image = copy.deepcopy(frame)

            if frame is not None:
                results = detector.process(frame)
            else:
                results = None

            if results and results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    preprocessed_landmarks = pre_process_landmarks(landmark_list)
                    preprocessed_point_history = gesture_history.preprocess_point_history(debug_image)
                    log_sample(number, mode, preprocessed_landmarks, preprocessed_point_history, KEYPOINT_CSV_PATH, POINT_HISTORY_CSV_PATH)

                    hand_sign_id = keypoint_classifier(preprocessed_landmarks)
                    finger_gesture_id, stabilized_finger_gesture_id = gesture_history.classify_finger_gesture(
                        point_history_classifier, preprocessed_point_history
                    )
                    gesture_history.update_point_history(landmark_list, hand_sign_id)

                    hand_sign_text = keypoint_labels[hand_sign_id] if hand_sign_id < len(keypoint_labels) else str(hand_sign_id)
                    finger_text = (
                        point_history_labels[stabilized_finger_gesture_id]
                        if stabilized_finger_gesture_id < len(point_history_labels)
                        else str(stabilized_finger_gesture_id)
                    )

                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text, finger_text)
            else:
                gesture_history.mark_no_hand()

            debug_image = draw_point_history(debug_image, gesture_history.point_history)
            debug_image = draw_info(debug_image, fps, mode, number, MODE_TEXT)

            cv.imshow('Hand Gesture Recognition', debug_image)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
