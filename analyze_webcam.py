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
output_csv = SCRIPT_DIR / "webcam_expression_user_neutral_baseline.csv"

# 사용자 3초 무표정 baseline을 저장해두면 디버깅할 때 편합니다.
user_neutral_baseline_output_csv = SCRIPT_DIR / "user_neutral_baseline.csv"

# 이제 neutral_distribution.csv는 필수로 쓰지 않습니다.
# anger가 neutral과 헷갈릴 때만 기존 anger label CSV를 보조 기준으로 사용합니다.
anger_distribution_path = SCRIPT_DIR / "anger_distribution.csv"

for required_path in [model_path, anger_distribution_path]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"필수 파일을 찾을 수 없습니다: {required_path}\n"
            "face_landmarker.task, anger_distribution.csv를 "
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

# 프로그램 시작 후 사용자의 무표정 얼굴을 수집하는 시간입니다.
CALIBRATION_SECONDS = 3.0

# 3초 동안 최소 이 정도 샘플은 잡혀야 baseline으로 인정합니다.
BASELINE_MIN_SAMPLES = 10

# 사용자 baseline의 std가 너무 작으면 거리 계산이 과하게 커질 수 있어 floor를 둡니다.
USER_STD_FLOOR = 0.015

# AI-Hub anger CSV의 std floor입니다.
AIHUB_STD_FLOOR = 0.02

# 사용자 baseline과의 z-distance가 이 값보다 작으면 neutral 후보로 봅니다.
USER_NEUTRAL_DISTANCE_THRESHOLD = 1.8

# baseline 대비 blendshape 평균 변화량이 이 값보다 작으면 neutral 후보로 봅니다.
USER_NEUTRAL_DELTA_ENERGY_THRESHOLD = 0.025

# MediaPipe 기반 점수가 이 값보다 낮으면 neutral 후보로 봅니다.
MEDIAPIPE_NEUTRAL_THRESHOLD = 15

# anger CSV가 이 점수 이상일 때만 anger 보조 판단을 강하게 반영합니다.
AIHUB_ANGER_SCORE_THRESHOLD = 58

# anger distance가 user neutral distance보다 어느 정도 더 가까워야 하는지 정합니다.
AIHUB_DISTANCE_MARGIN = 0.95

# anger 핵심 근육 변화량이 너무 작으면 anger로 보지 않습니다.
AIHUB_MIN_ANGER_CORE_DELTA = 0.03


# =========================
# 3. 비교에 사용할 주요 blendshape
# =========================

COMPARE_KEYS = [
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
    "browOuterUpLeft",
    "browOuterUpRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen"
]

ANGER_COMPARE_KEYS = [
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
# 6. CSV 분포 로드 / 거리 계산
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


anger_distribution = load_distribution_csv(anger_distribution_path)


def calculate_distribution_distance(data, distribution, keys, std_floor):
    squared_z_values = []

    for key in keys:
        if key not in distribution:
            continue

        value = float(data.get(key, 0.0))
        mean_value = distribution[key]["mean"]
        std_value = max(distribution[key]["std"], std_floor)

        z_value = (value - mean_value) / std_value
        squared_z_values.append(z_value ** 2)

    if len(squared_z_values) == 0:
        return None

    return float(np.sqrt(np.mean(squared_z_values)))


def calculate_delta_energy(raw_data, user_neutral_distribution, keys):
    delta_values = []

    for key in keys:
        if key not in user_neutral_distribution:
            continue

        value = float(raw_data.get(key, 0.0))
        baseline_mean = user_neutral_distribution[key]["mean"]

        delta_values.append(abs(value - baseline_mean))

    if len(delta_values) == 0:
        return 0.0

    return float(np.mean(delta_values))


def build_user_neutral_distribution(samples):
    all_keys = sorted(
        {
            key
            for sample in samples
            for key in sample.keys()
        }
    )

    distribution = {}

    for key in all_keys:
        values = np.array(
            [sample.get(key, 0.0) for sample in samples],
            dtype=np.float32
        )

        distribution[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return distribution


def save_distribution_csv(distribution, csv_path):
    rows = []

    for blendshape_name, stats in distribution.items():
        rows.append(
            {
                "blendshape": blendshape_name,
                "mean": stats["mean"],
                "std": stats["std"]
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def make_relative_blendshapes(raw_data, user_neutral_distribution):
    relative_data = {}
    delta_data = {}

    for key, value in raw_data.items():
        baseline_mean = user_neutral_distribution.get(
            key,
            {"mean": 0.0}
        )["mean"]

        delta = float(value) - baseline_mean

        # 감정 점수 계산에는 baseline보다 증가한 값만 사용합니다.
        # 예를 들어 원래 입꼬리가 약간 올라간 사람은 smile 기준값을 빼고 계산됩니다.
        positive_delta = max(delta, 0.0)

        relative_data[key] = positive_delta
        delta_data[key] = delta

    return relative_data, delta_data


def add_user_neutral_reference_scores(data, raw_blendshapes, user_neutral_distribution):
    user_neutral_distance = calculate_distribution_distance(
        raw_blendshapes,
        user_neutral_distribution,
        COMPARE_KEYS,
        USER_STD_FLOOR
    )

    user_neutral_delta_energy = calculate_delta_energy(
        raw_blendshapes,
        user_neutral_distribution,
        COMPARE_KEYS
    )

    if user_neutral_distance is None:
        user_neutral_score = 0
    else:
        user_neutral_score = 100 / (1 + user_neutral_distance)

    data["user_neutral_distance"] = user_neutral_distance
    data["user_neutral_delta_energy"] = user_neutral_delta_energy
    data["user_neutral_score"] = user_neutral_score

    return data


def add_aihub_anger_reference_scores(data, raw_blendshapes):
    user_neutral_distance = data.get("user_neutral_distance", None)

    aihub_anger_distance = calculate_distribution_distance(
        raw_blendshapes,
        anger_distribution,
        ANGER_COMPARE_KEYS,
        AIHUB_STD_FLOOR
    )

    data["aihub_anger_distance"] = aihub_anger_distance

    if user_neutral_distance is None or aihub_anger_distance is None:
        data["aihub_anger_score"] = 0
        data["aihub_reference_emotion"] = "Unknown"
        return data

    total_distance = user_neutral_distance + aihub_anger_distance

    if total_distance == 0:
        anger_score = 50
    else:
        # neutral에서 멀고 anger CSV에는 가까울수록 anger score가 커집니다.
        anger_score = user_neutral_distance / total_distance * 100

    data["aihub_anger_score"] = anger_score

    if aihub_anger_distance < user_neutral_distance * AIHUB_DISTANCE_MARGIN:
        data["aihub_reference_emotion"] = "Anger"
    else:
        data["aihub_reference_emotion"] = "UserNeutral"

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
# 7. 사용자 baseline 기준 감정 계산 함수
# =========================

def calculate_emotions(relative_data):
    mouth_smile_left = relative_data.get("mouthSmileLeft", 0)
    mouth_smile_right = relative_data.get("mouthSmileRight", 0)

    mouth_frown_left = relative_data.get("mouthFrownLeft", 0)
    mouth_frown_right = relative_data.get("mouthFrownRight", 0)

    mouth_press_left = relative_data.get("mouthPressLeft", 0)
    mouth_press_right = relative_data.get("mouthPressRight", 0)

    brow_inner_up = relative_data.get("browInnerUp", 0)
    brow_down_left = relative_data.get("browDownLeft", 0)
    brow_down_right = relative_data.get("browDownRight", 0)

    brow_outer_up_left = relative_data.get("browOuterUpLeft", 0)
    brow_outer_up_right = relative_data.get("browOuterUpRight", 0)

    eye_squint_left = relative_data.get("eyeSquintLeft", 0)
    eye_squint_right = relative_data.get("eyeSquintRight", 0)

    eye_wide_left = relative_data.get("eyeWideLeft", 0)
    eye_wide_right = relative_data.get("eyeWideRight", 0)

    nose_sneer_left = relative_data.get("noseSneerLeft", 0)
    nose_sneer_right = relative_data.get("noseSneerRight", 0)

    jaw_open = relative_data.get("jawOpen", 0)

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

    emotion_scores = {}

    emotion_scores["joy_raw"] = joy_raw
    emotion_scores["sadness_raw"] = sadness_raw
    emotion_scores["anger_raw"] = anger_raw
    emotion_scores["surprise_raw"] = surprise_raw

    emotion_scores["joy"] = raw_to_percent(joy_raw, "joy")
    emotion_scores["sadness"] = raw_to_percent(sadness_raw, "sadness")
    emotion_scores["anger"] = raw_to_percent(anger_raw, "anger")
    emotion_scores["surprise"] = raw_to_percent(surprise_raw, "surprise")

    return emotion_scores


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

    user_neutral_distance = data.get("user_neutral_distance", None)
    user_neutral_delta_energy = data.get("user_neutral_delta_energy", 0)

    aihub_anger_distance = data.get("aihub_anger_distance", None)
    aihub_anger_score = data.get("aihub_anger_score", 0)

    anger_core_delta = data.get("anger_core_delta", 0)
    data["anger_core_delta"] = anger_core_delta

    is_user_neutral = (
        user_neutral_distance is not None and
        user_neutral_distance <= USER_NEUTRAL_DISTANCE_THRESHOLD and
        user_neutral_delta_energy <= USER_NEUTRAL_DELTA_ENERGY_THRESHOLD
    )

    if user_neutral_distance is not None and aihub_anger_distance is not None:
        anger_csv_confirms = (
            aihub_anger_distance < user_neutral_distance * AIHUB_DISTANCE_MARGIN and
            aihub_anger_score >= AIHUB_ANGER_SCORE_THRESHOLD and
            anger_core_delta >= AIHUB_MIN_ANGER_CORE_DELTA
        )
    else:
        anger_csv_confirms = False

    data["is_user_neutral"] = is_user_neutral
    data["anger_csv_confirms"] = anger_csv_confirms

    if is_user_neutral or mediapipe_percent < MEDIAPIPE_NEUTRAL_THRESHOLD:
        if anger_csv_confirms:
            data["final_source"] = "user_neutral_baseline+anger_csv"
            return "Anger", max(30, aihub_anger_score), data

        data["final_source"] = "user_neutral_baseline"
        return "Neutral", data.get("user_neutral_score", 0), data

    if mediapipe_emotion == "Anger":
        if anger_csv_confirms:
            data["final_source"] = "relative_blendshape+anger_csv"
            return "Anger", max(mediapipe_percent, aihub_anger_score), data

        non_anger_emotions = {
            "Joy": data["joy"],
            "Sadness": data["sadness"],
            "Surprise": data["surprise"]
        }

        second_emotion = max(non_anger_emotions, key=non_anger_emotions.get)
        second_percent = non_anger_emotions[second_emotion]

        if second_percent >= MEDIAPIPE_NEUTRAL_THRESHOLD:
            data["final_source"] = "relative_blendshape_anger_rejected"
            return second_emotion, second_percent, data

        data["final_source"] = "anger_rejected_as_neutral"
        return "Neutral", data.get("user_neutral_score", 0), data

    # Joy, Sadness, Surprise는 사용자 baseline 대비 변화량만으로 판단합니다.
    # 단, MediaPipe 점수가 낮고 anger CSV가 강하게 확인될 때만 숨은 anger로 보정합니다.
    if anger_csv_confirms and mediapipe_percent < 35:
        data["final_source"] = "anger_csv_override_low_confidence"
        return "Anger", max(mediapipe_percent, aihub_anger_score), data

    data["final_source"] = "relative_blendshape"
    return mediapipe_emotion, mediapipe_percent, data


# =========================
# 8. 화면 표시 함수
# =========================

def draw_emotion_box(frame, box, emotion, percent):
    if box is None:
        return

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


def collect_user_neutral_baseline(cap):
    print("\n[Baseline] 3초 동안 무표정으로 화면을 바라봐 주세요.")
    print("[Baseline] 이 값이 현재 사용자의 neutral label 데이터로 사용됩니다.")

    samples = []
    calibration_start_time = time.time()
    calibration_frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[Baseline] 웹캠 프레임을 읽을 수 없습니다.")
            break

        calibration_frame_idx += 1
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        elapsed = time.time() - calibration_start_time
        remaining = max(0, CALIBRATION_SECONDS - elapsed)

        if calibration_frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
            face_landmarks, blendshapes = detect_frame(frame)

            if face_landmarks is not None and blendshapes is not None:
                raw_blendshapes = extract_blendshape_dict(blendshapes)
                samples.append(raw_blendshapes)

                box = get_face_box(face_landmarks, width, height)
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 255, 255),
                    2
                )

        draw_status_text(
            frame,
            "Neutral calibration: keep a neutral face",
            35,
            (255, 255, 255)
        )

        draw_status_text(
            frame,
            f"Remaining: {remaining:.1f}s | samples: {len(samples)}",
            70,
            (255, 255, 255)
        )

        cv2.imshow("Real-Time Webcam Expression Analysis", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[Baseline] q 입력됨. 프로그램을 종료합니다.")
            return None

        if elapsed >= CALIBRATION_SECONDS:
            break

    if len(samples) < BASELINE_MIN_SAMPLES:
        print(
            f"[Baseline] baseline 샘플이 부족합니다. "
            f"수집 샘플: {len(samples)}, 필요 샘플: {BASELINE_MIN_SAMPLES}"
        )
        return None

    user_neutral_distribution = build_user_neutral_distribution(samples)
    save_distribution_csv(
        user_neutral_distribution,
        user_neutral_baseline_output_csv
    )

    print("[Baseline] 사용자 neutral baseline 저장 완료!")
    print("[Baseline] 저장 위치:", user_neutral_baseline_output_csv)
    print("[Baseline] 수집 샘플 수:", len(samples))

    return user_neutral_distribution


cap, camera_index = open_first_available_camera(max_camera_index=10)

if cap is None:
    print("카메라를 열 수 없습니다. 카메라 연결 상태와 권한을 확인하세요.")
    exit()

user_neutral_distribution = collect_user_neutral_baseline(cap)

if user_neutral_distribution is None:
    cap.release()
    cv2.destroyAllWindows()
    exit()

print("\n웹캠 실시간 분석 시작!")
print("사용자 neutral baseline:", user_neutral_baseline_output_csv.name)
print("anger 보조 CSV:", anger_distribution_path.name)
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
last_source = "user_neutral_baseline"

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

                relative_blendshapes, delta_blendshapes = make_relative_blendshapes(
                    raw_blendshapes,
                    user_neutral_distribution
                )

                for key, value in raw_blendshapes.items():
                    data[key] = value

                for key, value in delta_blendshapes.items():
                    data[f"delta_{key}"] = value

                emotion_scores = calculate_emotions(relative_blendshapes)

                for key, value in emotion_scores.items():
                    data[key] = value

                data["anger_core_delta"] = calculate_anger_core(relative_blendshapes)
                data["anger_core_raw"] = calculate_anger_core(raw_blendshapes)

                data = add_user_neutral_reference_scores(
                    data,
                    raw_blendshapes,
                    user_neutral_distribution
                )

                data = add_aihub_anger_reference_scores(
                    data,
                    raw_blendshapes
                )

                emotion, percent, data = get_dominant_emotion(data)

                last_box = get_face_box(face_landmarks, width, height)
                last_emotion = emotion
                last_percent = percent
                last_source = data.get("final_source", "relative_blendshape")

                data["dominant_emotion"] = emotion
                data["dominant_percent"] = percent

                results.append(data)

        if last_box is not None:
            draw_emotion_box(frame, last_box, last_emotion, last_percent)

        draw_status_text(
            frame,
            f"q: finish | source: {last_source}",
            height - 60,
            (255, 255, 255)
        )

        draw_status_text(
            frame,
            f"emotion: {last_emotion} {last_percent:.0f}%",
            height - 30,
            get_emotion_color(last_emotion)
        )

        cv2.imshow("Real-Time Webcam Expression Analysis", frame)

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
        "user_neutral_score",
        "user_neutral_distance",
        "user_neutral_delta_energy",
        "aihub_anger_score",
        "aihub_anger_distance",
        "anger_core_delta",
        "mediapipe_label",
        "mediapipe_percent",
        "is_user_neutral",
        "anger_csv_confirms",
        "final_source",
        "dominant_emotion",
        "dominant_percent"
    ]

    existing_cols = [col for col in emotion_cols if col in df.columns]

    print("\n결과 요약:")
    print(df[existing_cols].tail())

    print("\n최종 감정 분류 개수:")
    print(df["dominant_emotion"].value_counts())

    if "final_source" in df.columns:
        print("\n최종 판단 출처 개수:")
        print(df["final_source"].value_counts())
