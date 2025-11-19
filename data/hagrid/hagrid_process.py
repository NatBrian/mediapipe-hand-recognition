import os
import csv
import cv2
from tqdm import tqdm
import mediapipe as mp

# ----------------------------
# 配置
# ----------------------------
# 图片所在根目录
IMAGE_ROOT_DIR = r"D:\GitHubRepo\mediapipe-hand-recognition\data\hagrid\pics\hagrid-sample-30k-384p\hagrid_30k"

# CSV 保存路径
CSV_DIR = r"D:\GitHubRepo\mediapipe-hand-recognition\data\hagrid"
os.makedirs(CSV_DIR, exist_ok=True)
KEYPOINT_CSV = os.path.join(CSV_DIR, "hagrid_keypoint.csv")
LABEL_CSV = os.path.join(CSV_DIR, "hagrid_keypoint_labels.csv")

# HaGRID 类别
CLASS_NAMES = [
    'call','no_gesture','dislike','fist','four','like','mute','ok','one','palm',
    'peace','peace_inverted','rock','stop','stop_inverted','three','three2','two_up','two_up_inverted'
]
CLASS_MAP = {name:i for i,name in enumerate(CLASS_NAMES)}

# ----------------------------
# MediaPipe Hands 初始化
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ----------------------------
# 遍历图片文件夹并生成 CSV
# ----------------------------
with open(KEYPOINT_CSV,'w',newline='') as kp_file:
    kp_writer = csv.writer(kp_file)

    # 遍历所有类别文件夹
    for class_name in CLASS_NAMES:
        class_dir_pattern = os.path.join(IMAGE_ROOT_DIR, f"*{class_name}*")
        matched_dirs = [d for d in os.listdir(IMAGE_ROOT_DIR) if class_name in d.lower()]
        if not matched_dirs:
            print(f"Warning: No folder matched for class '{class_name}', skipping...")
            continue

        for folder_name in matched_dirs:
            folder_path = os.path.join(IMAGE_ROOT_DIR, folder_name)
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]

            for img_name in tqdm(img_files, desc=f"Processing {class_name}"):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])

                    kp_writer.writerow([CLASS_MAP[class_name]] + keypoints)

# ----------------------------
# 生成标签 CSV
# ----------------------------
with open(LABEL_CSV,'w',newline='') as f:
    writer = csv.writer(f)
    for cls_name, cls_id in CLASS_MAP.items():
        writer.writerow([cls_id, cls_name])

print("Finished generating hagrid_keypoint.csv and hagrid_keypoint_labels.csv")
