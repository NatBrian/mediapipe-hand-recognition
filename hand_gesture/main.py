from __future__ import annotations

import argparse
import copy

import cv2 as cv

from hand_gesture.utils.smoother import EMASmoother, KalmanSmoother

from hand_gesture import PACKAGE_ROOT, REPO_ROOT
from hand_gesture.application import ShortcutExecutor
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
# 选择要加载的 keypoint tflite 模型：change KEYPOINT_MODEL_PATH in main.py and model_pyth in classifier.py 
# - `keypoint_classifier_1dcnn.tflite` : 1D-CNN，训练时使用 `Input(shape=(21,2))`，推理需要 `(1,21,2)`
# - `keypoint_classifier_mlp.tflite`   : MLP / 扁平输入，通常接受 `(1,42)`
# 默认指向仓库中存在的 1D-CNN 模型（可根据需要改回 MLP）
KEYPOINT_MODEL_PATH = REPO_ROOT / 'models/keypoint/keypoint_classifier_mlp.tflite'
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
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition with optional application shortcuts')
    parser.add_argument(
        '--application', '-a',
        action='store_true',
        help='Enable application mode: gestures will trigger keyboard shortcuts and mouse control'
    )
    args = parser.parse_args()

    keypoint_labels = load_labels(KEYPOINT_LABEL_PATH)
    point_history_labels = load_labels(POINT_HISTORY_LABEL_PATH)

    keypoint_classifier = KeyPointClassifier(model_path=KEYPOINT_MODEL_PATH)
    point_history_classifier = PointHistoryClassifier(model_path=POINT_HISTORY_MODEL_PATH)

    fps_calc = CvFpsCalc(buffer_len=10)
    gesture_history = GestureHistory()
    
    # Only enable shortcuts if --application flag is provided
    if args.application:
        shortcut_executor = ShortcutExecutor(debounce_seconds=0.3, enabled=True)
        print("[Application Mode] Shortcuts and mouse control ENABLED")
    else:
        shortcut_executor = ShortcutExecutor(debounce_seconds=0.3, enabled=False)
        print("[Normal Mode] Shortcuts and mouse control DISABLED (use --application to enable)")

    #smoother
    ema_smoother = EMASmoother(alpha=0.6)
    kalman_smoother = KalmanSmoother(dt=1.0, process_variance=1e-2, measurement_variance=1e-1)

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

                    # 1. 取 index finger (id = 8)
                    # -------------------------------
                    raw_x, raw_y = landmark_list[8]

                    # -------------------------------
                    # 2. 平滑处理（任选其一）
                    # -------------------------------
                    ema_x, ema_y = ema_smoother.smooth((raw_x, raw_y))
                    kal_x, kal_y = kalman_smoother.smooth((raw_x, raw_y))

                    # 建议用 Kalman（更稳）
                    smoothed_point = (kal_x, kal_y)

                    # -------------------------------
                    # 3. 用平滑后的点替换 landmark_list 中第 8 个点
                    # -------------------------------
                    landmark_list[8][0] = int(smoothed_point[0])
                    landmark_list[8][1] = int(smoothed_point[1])

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

                    # Execute shortcuts only in NORMAL mode and if application mode is enabled
                    if mode == LoggingMode.NORMAL and shortcut_executor.enabled:
                        # Move mouse if Pointer gesture is detected
                        if hand_sign_id == 2:  # Pointer gesture ID
                            shortcut_executor.move_mouse(landmark_list, CAPTURE_WIDTH, CAPTURE_HEIGHT)
                        else:
                            shortcut_executor.execute_keypoint_gesture(hand_sign_id)
                        # Point history gestures disabled for now - uncomment to enable
                        # shortcut_executor.execute_point_history_gesture(stabilized_finger_gesture_id)

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
