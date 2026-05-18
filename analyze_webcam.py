import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import platform
import time
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent

model_path = SCRIPT_DIR / "face_landmarker.task"
output_csv = SCRIPT_DIR / "webcam_expression_mediapipe_with_aihub_csv.csv"

neutral_distribution_path = SCRIPT_DIR / "neutral_distribution.csv"
anger_distribution_path = SCRIPT_DIR / "anger_distribution.csv"

for required_path in [model_path, neutral_distribution_path, anger_distribution_path]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"필수 파일을 찾을 수 없습니다: {required_path}\n"
            "face_landmarker.task, neutral_distribution.csv, anger_distribution.csv를 "
            "이 .py 파일과 같은 폴더에 넣어주세요."
        )


# =========================
# 2. 감정 민감도 / 설정값
# =========================

SENSITIVITY = {
    "joy": 4.0,
    "sadness": 6.0,
    "anger": 4.0,
    "surprise": 4.5
}

ANALYZE_EVERY_N_FRAMES = 3

STD_FLOOR = 0.02
AIHUB_ANGER_SCORE_THRESHOLD = 58
AIHUB_DISTANCE_MARGIN = 0.95
AIHUB_MIN_ANGER_CORE = 0.03

# MediaPipe 점수가 이 값보다 낮으면 일단 Neutral 후보로 봄
MEDIAPIPE_NEUTRAL_THRESHOLD = 15


# =========================
# 3. AI-Hub 기준 분포 비교에 사용할 blendshape
# =========================

AIHUB_COMPARE_KEYS = [
    "browDownLeft",
    "browDownRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthPressLeft",
    "mouthPressRight",
    "noseSneerLeft",
    "noseSneerRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "browInnerUp",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen"
]


# =========================
# 4. MediaPipe Face Landmarker 설정
# =========================

base_options = python.BaseOptions(model_asset_path=str(model_path))

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# 5. 유틸 함수
# =========================

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value, emotion_name):
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)
    adjusted = raw_value * sensitivity

    percent = (adjusted / (adjusted + 1)) * 100
    return clamp(percent)


def get_emotion_color(emotion):
    colors = {
        "Joy": (0, 255, 255),
        "Sadness": (255, 0, 0),
        "Anger": (0, 0, 255),
        "Surprise": (255, 0, 255),
        "Neutral": (255, 255, 255)
    }

    return colors.get(emotion, (255, 255, 255))


def extract_blendshape_dict(blendshapes):
    data = {}

    for category in blendshapes:
        data[category.category_name] = float(category.score)

    return data


def get_face_box(face_landmarks, frame_width, frame_height):
    x_values = []
    y_values = []

    for landmark in face_landmarks:
        x_values.append(int(landmark.x * frame_width))
        y_values.append(int(landmark.y * frame_height))

    x_min = max(min(x_values) - 25, 0)
    y_min = max(min(y_values) - 35, 0)
    x_max = min(max(x_values) + 25, frame_width)
    y_max = min(max(y_values) + 25, frame_height)

    return x_min, y_min, x_max, y_max


def detect_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks or not detection_result.face_blendshapes:
        return None, None

    return detection_result.face_landmarks[0], detection_result.face_blendshapes[0]


# =========================
# 6. AI-Hub 기준 분포 CSV 로드 / 비교 함수
# =========================

def load_distribution_csv(csv_path):
    distribution_df = pd.read_csv(csv_path)

    required_columns = {"blendshape", "mean", "std"}
    missing_columns = required_columns - set(distribution_df.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} 파일에 필요한 컬럼이 없습니다: {missing_columns}\n"
            "필요 컬럼: blendshape, mean, std"
        )

    distribution = {}

    for _, row in distribution_df.iterrows():
        blendshape_name = row["blendshape"]

        distribution[blendshape_name] = {
            "mean": float(row["mean"]),
            "std": float(row["std"])
        }

    return distribution


neutral_distribution = load_distribution_csv(neutral_distribution_path)
anger_distribution = load_distribution_csv(anger_distribution_path)


def calculate_distribution_distance(data, distribution):
    squared_z_values = []

    for key in AIHUB_COMPARE_KEYS:
        if key not in distribution:
            continue

        value = float(data.get(key, 0.0))
        mean_value = distribution[key]["mean"]
        std_value = max(distribution[key]["std"], STD_FLOOR)

        z_value = (value - mean_value) / std_value
        squared_z_values.append(z_value ** 2)

    if len(squared_z_values) == 0:
        return None

    return float(np.sqrt(np.mean(squared_z_values)))


def add_aihub_reference_scores(data):
    neutral_distance = calculate_distribution_distance(data, neutral_distribution)
    anger_distance = calculate_distribution_distance(data, anger_distribution)

    if neutral_distance is None or anger_distance is None:
        data["aihub_neutral_distance"] = None
        data["aihub_anger_distance"] = None
        data["aihub_neutral_score"] = 0
        data["aihub_anger_score"] = 0
        data["aihub_reference_emotion"] = "Unknown"
        return data

    total_distance = neutral_distance + anger_distance

    if total_distance == 0:
        neutral_score = 50
        anger_score = 50
    else:
        neutral_score = anger_distance / total_distance * 100
        anger_score = neutral_distance / total_distance * 100

    data["aihub_neutral_distance"] = neutral_distance
    data["aihub_anger_distance"] = anger_distance
    data["aihub_neutral_score"] = neutral_score
    data["aihub_anger_score"] = anger_score

    if anger_distance < neutral_distance:
        data["aihub_reference_emotion"] = "Anger"
    else:
        data["aihub_reference_emotion"] = "Neutral"

    return data


def calculate_anger_core(data):
    anger_core = (
        data.get("browDownLeft", 0.0) +
        data.get("browDownRight", 0.0) +
        data.get("eyeSquintLeft", 0.0) +
        data.get("eyeSquintRight", 0.0) +
        data.get("mouthPressLeft", 0.0) +
        data.get("mouthPressRight", 0.0)
    ) / 6

    return anger_core


# =========================
# 7. MediaPipe 기반 감정 계산 함수
# =========================

def calculate_emotions(data):
    mouth_smile_left = data.get("mouthSmileLeft", 0)
    mouth_smile_right = data.get("mouthSmileRight", 0)

    mouth_frown_left = data.get("mouthFrownLeft", 0)
    mouth_frown_right = data.get("mouthFrownRight", 0)

    mouth_press_left = data.get("mouthPressLeft", 0)
    mouth_press_right = data.get("mouthPressRight", 0)

    brow_inner_up = data.get("browInnerUp", 0)
    brow_down_left = data.get("browDownLeft", 0)
    brow_down_right = data.get("browDownRight", 0)

    brow_outer_up_left = data.get("browOuterUpLeft", 0)
    brow_outer_up_right = data.get("browOuterUpRight", 0)

    eye_squint_left = data.get("eyeSquintLeft", 0)
    eye_squint_right = data.get("eyeSquintRight", 0)

    eye_wide_left = data.get("eyeWideLeft", 0)
    eye_wide_right = data.get("eyeWideRight", 0)

    nose_sneer_left = data.get("noseSneerLeft", 0)
    nose_sneer_right = data.get("noseSneerRight", 0)

    jaw_open = data.get("jawOpen", 0)

    smile_avg = (mouth_smile_left + mouth_smile_right) / 2
    eye_wide_avg = (eye_wide_left + eye_wide_right) / 2
    mouth_press_avg = (mouth_press_left + mouth_press_right) / 2
    brow_outer_up_avg = (brow_outer_up_left + brow_outer_up_right) / 2
    brow_up_avg = (brow_inner_up + brow_outer_up_avg) / 2

    joy_raw = (
        mouth_smile_left * 0.45 +
        mouth_smile_right * 0.45 +
        eye_squint_left * 0.05 +
        eye_squint_right * 0.05
    )

    sadness_raw = (
        mouth_frown_left * 0.30 +
        mouth_frown_right * 0.30 +
        brow_inner_up * 0.35 +
        mouth_press_left * 0.025 +
        mouth_press_right * 0.025
    )

    anger_raw = (
        brow_down_left * 0.23 +
        brow_down_right * 0.23 +
        eye_squint_left * 0.13 +
        eye_squint_right * 0.13 +
        nose_sneer_left * 0.10 +
        nose_sneer_right * 0.10 +
        mouth_press_left * 0.04 +
        mouth_press_right * 0.04
    )

    if smile_avg < 0.15 and eye_wide_avg > 0.18:
        anger_raw += eye_wide_avg * 0.35

    if mouth_press_avg > 0.15:
        anger_raw += mouth_press_avg * 0.25

    if not (smile_avg < 0.15 and eye_wide_avg > 0.18):
        anger_raw = anger_raw * (1 - jaw_open * 0.4)

    anger_signal = (
        brow_down_left + brow_down_right +
        eye_squint_left + eye_squint_right +
        nose_sneer_left + nose_sneer_right +
        mouth_press_left + mouth_press_right
    ) / 8

    surprise_raw = (
        jaw_open * 0.30 +
        brow_up_avg * 0.30 +
        eye_wide_left * 0.20 +
        eye_wide_right * 0.20
    )

    surprise_raw *= (1 - anger_signal * 0.5)

    if jaw_open > 0.4 and brow_inner_up < 0.15 and eye_wide_avg < 0.15:
        surprise_raw *= 0.4

    if smile_avg < 0.15 and eye_wide_avg > 0.18 and jaw_open < 0.25:
        surprise_raw *= 0.5

    data["joy_raw"] = joy_raw
    data["sadness_raw"] = sadness_raw
    data["anger_raw"] = anger_raw
    data["surprise_raw"] = surprise_raw

    data["joy"] = raw_to_percent(joy_raw, "joy")
    data["sadness"] = raw_to_percent(sadness_raw, "sadness")
    data["anger"] = raw_to_percent(anger_raw, "anger")
    data["surprise"] = raw_to_percent(surprise_raw, "surprise")

    return data


def get_dominant_emotion(data):
    emotions = {
        "Joy": data["joy"],
        "Sadness": data["sadness"],
        "Anger": data["anger"],
        "Surprise": data["surprise"]
    }

    mediapipe_emotion = max(emotions, key=emotions.get)
    mediapipe_percent = emotions[mediapipe_emotion]

    data["mediapipe_label"] = mediapipe_emotion
    data["mediapipe_percent"] = mediapipe_percent

    aihub_neutral_distance = data.get("aihub_neutral_distance", None)
    aihub_anger_distance = data.get("aihub_anger_distance", None)
    aihub_anger_score = data.get("aihub_anger_score", 0)
    aihub_neutral_score = data.get("aihub_neutral_score", 0)

    anger_core = calculate_anger_core(data)
    data["anger_core"] = anger_core

    if aihub_neutral_distance is not None and aihub_anger_distance is not None:
        anger_is_closer = aihub_anger_distance < aihub_neutral_distance * AIHUB_DISTANCE_MARGIN
    else:
        anger_is_closer = False

    final_source = "mediapipe"

    if mediapipe_percent < MEDIAPIPE_NEUTRAL_THRESHOLD:
        if (
            anger_is_closer and
            aihub_anger_score >= AIHUB_ANGER_SCORE_THRESHOLD and
            anger_core >= AIHUB_MIN_ANGER_CORE
        ):
            data["final_source"] = "csv_label_aux"
            return "Anger", max(30, aihub_anger_score), data

        data["final_source"] = "csv_label_aux"
        return "Neutral", aihub_neutral_score, data

    if mediapipe_emotion == "Anger":
        data["final_source"] = "mediapipe+csv_score"
        return "Anger", max(mediapipe_percent, aihub_anger_score), data

    if (
        anger_is_closer and
        aihub_anger_score >= 65 and
        anger_core >= AIHUB_MIN_ANGER_CORE
    ):
        data["final_source"] = "csv_label_aux"
        return "Anger", max(mediapipe_percent, aihub_anger_score), data

    data["final_source"] = final_source
    return mediapipe_emotion, mediapipe_percent, data


# =========================
# 8. 화면 표시 함수
# =========================

def draw_emotion_box(frame, box, emotion, percent):
    if box is None:
        return

    # 녹화 영상 코드와 동일하게 Neutral이면 박스를 표시하지 않음
    if emotion == "Neutral":
        return

    x_min, y_min, x_max, y_max = box
    box_color = get_emotion_color(emotion)

    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        box_color,
        3
    )

    text = f"{emotion} {percent:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    text_x = x_min
    text_y = max(y_min - 12, text_height + 12)

    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - 10),
        (text_x + text_width + 12, text_y + 6),
        box_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (text_x + 6, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )


def draw_status_text(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# =========================
# 9. 웹캠 열기
# =========================

def create_video_capture(camera_index):
    if platform.system() == "Windows":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    return cv2.VideoCapture(camera_index)


def open_first_available_camera(max_camera_index=10):
    for camera_index in range(max_camera_index + 1):
        cap = create_video_capture(camera_index)

        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            ret, frame = cap.read()

            if ret and frame is not None:
                print(f"{camera_index}번 카메라를 사용합니다.")
                return cap, camera_index

        cap.release()

    return None, None


cap, camera_index = open_first_available_camera(max_camera_index=10)

if cap is None:
    print("카메라를 열 수 없습니다. 카메라 연결 상태와 권한을 확인하세요.")
    exit()

print("웹캠 실시간 분석 시작!")
print("기준 CSV:", neutral_distribution_path.name, anger_distribution_path.name)
print("q를 누르면 종료하고 CSV로 저장합니다.")


# =========================
# 10. 실시간 웹캠 분석 루프
# =========================

results = []
frame_idx = 0
start_time = time.time()

last_box = None
last_emotion = "Neutral"
last_percent = 0
last_source = "mediapipe"

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        frame_idx += 1
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
            face_landmarks, blendshapes = detect_frame(frame)

            if face_landmarks is not None and blendshapes is not None:
                data = {
                    "frame": frame_idx,
                    "time": time.time() - start_time
                }

                raw_blendshapes = extract_blendshape_dict(blendshapes)

                for key, value in raw_blendshapes.items():
                    data[key] = value

                data = calculate_emotions(data)
                data = add_aihub_reference_scores(data)

                emotion, percent, data = get_dominant_emotion(data)

                last_box = get_face_box(face_landmarks, width, height)
                last_emotion = emotion
                last_percent = percent
                last_source = data.get("final_source", "mediapipe")

                data["dominant_emotion"] = emotion
                data["dominant_percent"] = percent

                results.append(data)

        if last_box is not None:
            draw_emotion_box(frame, last_box, last_emotion, last_percent)

        draw_status_text(
            frame,
            f"q: finish | source: {last_source}",
            height - 30,
            (255, 255, 255)
        )

        cv2.imshow("Real-Time Webcam Expression Analysis with AI-Hub CSV Reference", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("q 입력됨. 분석을 종료합니다.")
            break

except KeyboardInterrupt:
    print("\nCtrl+C 입력됨. 분석을 종료하고 CSV 저장 단계로 이동합니다.")

finally:
    cap.release()
    cv2.destroyAllWindows()


# =========================
# 11. CSV 저장
# =========================

if len(results) == 0:
    print("저장할 데이터가 없습니다. 얼굴이 웹캠에 잘 보이는지 확인하세요.")

else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))

    emotion_cols = [
        "joy",
        "sadness",
        "anger",
        "surprise",
        "aihub_neutral_score",
        "aihub_anger_score",
        "aihub_neutral_distance",
        "aihub_anger_distance",
        "mediapipe_label",
        "mediapipe_percent",
        "final_source",
        "dominant_emotion",
        "dominant_percent"
    ]

    existing_cols = [col for col in emotion_cols if col in df.columns]

    print("\n결과 요약:")
    print(df[existing_cols].tail())

    print("\n최종 감정 분류 개수:")
    print(df["dominant_emotion"].value_counts())

    if "aihub_reference_emotion" in df.columns:
        print("\nAI-Hub 기준 감정 분류 개수:")
        print(df["aihub_reference_emotion"].value_counts())

    if "final_source" in df.columns:
        print("\n최종 판단 출처 개수:")
        print(df["final_source"].value_counts())
