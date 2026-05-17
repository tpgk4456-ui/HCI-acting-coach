import ast
import subprocess
import threading
import time
import platform
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent

# face_landmarker.task는 이 .py 파일과 같은 폴더에 둡니다.
model_path = SCRIPT_DIR / "face_landmarker.task"

# 최종 CSV 저장 위치
output_csv = SCRIPT_DIR / "user_expression_custom_with_aihub_aux.csv"

# AI-Hub Docker 모델에 넘길 얼굴 crop 이미지 저장 위치
# Docker 컨테이너에서 /input/face.jpg로 읽을 수 있도록
# 이 폴더를 Docker 실행 시 /input으로 마운트해야 합니다.
crop_dir = SCRIPT_DIR / "aihub_input"
crop_dir.mkdir(exist_ok=True)

CROP_PATH_WINDOWS = str(crop_dir / "face.jpg")
CROP_PATH_DOCKER = "/input/face.jpg"

# Docker 컨테이너 이름
DOCKER_CONTAINER = "facialemotion_runner"


if not model_path.exists():
    raise FileNotFoundError(
        f"face_landmarker.task 파일을 찾을 수 없습니다: {model_path}\n"
        "face_landmarker.task 파일을 현재 .py 파일과 같은 폴더에 넣어주세요."
    )


# =========================
# 2. MediaPipe 감정 민감도 / 설정값
# =========================

SENSITIVITY = {
    "joy": 4.0,
    "sadness": 6.0,
    "anger": 4.0,
    "surprise": 4.5
}

CALIBRATION_SECONDS = 3.0
ANALYZE_EVERY_N_FRAMES = 3

NOISE_FLOOR = 0.015
STD_MULTIPLIER = 2.0

NEUTRAL_THRESHOLD = 18
SMOOTHING_ALPHA = 0.65


# =========================
# 3. AI-Hub 보조 판단 설정
# =========================

# AI-Hub 모델을 몇 초마다 호출할지
# 너무 짧으면 웹캠이 끊길 수 있습니다.
AIHUB_PREDICTION_INTERVAL_SECONDS = 1.3

# AI-Hub confidence가 이 값보다 낮으면 보조 판단에 사용하지 않음
AIHUB_MIN_CONFIDENCE_TO_USE = 45.0

# AI-Hub baseline 대비 이 정도 이상 증가해야 보조 판단에 사용
AIHUB_DELTA_THRESHOLD = 20.0

# MediaPipe 결과가 이 값 이상이면 충분히 확신한다고 보고 AI-Hub가 덮어쓰지 않음
MEDIAPIPE_STRONG_THRESHOLD = 30.0

# AI-Hub가 보조로 개입할 수 있는 감정
AIHUB_AUX_EMOTIONS = [
    "Joy",
    "Anger",
    "Sadness",
    "Surprise",
    "Anxiety",
    "Hurt"
]


# =========================
# 4. AI-Hub 라벨 / 색상 매핑
# =========================

LABEL_MAP = {
    "기쁨": "Joy",
    "분노": "Anger",
    "슬픔": "Sadness",
    "당황": "Surprise",
    "중립": "Neutral",
    "불안": "Anxiety",
    "상처": "Hurt"
}

COLOR_MAP = {
    "Joy": (0, 255, 255),
    "Anger": (0, 0, 255),
    "Sadness": (255, 0, 0),
    "Surprise": (255, 0, 255),
    "Neutral": (255, 255, 255),
    "Anxiety": (0, 165, 255),
    "Hurt": (255, 255, 0)
}


# =========================
# 5. AI-Hub 공유 상태값
# =========================

aihub_latest_raw_label = "Neutral"
aihub_latest_raw_percent = 0.0
aihub_latest_probs = {}
aihub_latest_delta_probs = {}
aihub_latest_aux_label = "Neutral"
aihub_latest_aux_percent = 0.0
aihub_latest_time = 0.0

aihub_baseline_probs = None
aihub_baseline_samples = []

is_aihub_predicting = False
last_aihub_prediction_time = 0.0

aihub_lock = threading.Lock()


# =========================
# 6. MediaPipe Face Landmarker 설정
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
# 7. 웹캠 선택
# =========================

def create_video_capture(camera_index):
    if platform.system() == "Windows":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    return cv2.VideoCapture(camera_index)


def open_available_camera(max_camera_index=10):
    for camera_index in range(max_camera_index + 1):
        print(f"{camera_index}번 카메라 확인 중...")

        cap = create_video_capture(camera_index)

        if not cap.isOpened():
            cap.release()
            continue

        valid_frame = None

        for _ in range(10):
            ret, frame = cap.read()

            if ret and frame is not None:
                valid_frame = frame
                break

        if valid_frame is None:
            cap.release()
            continue

        print(f"{camera_index}번 카메라가 감지되었습니다.")
        print("이 카메라를 사용하려면 y, 다음 카메라를 보려면 n, 종료하려면 q를 누르세요.")

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            preview_frame = cv2.flip(frame, 1)

            cv2.putText(
                preview_frame,
                f"Camera {camera_index} | y: use | n: next | q: quit",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Camera Selection", preview_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("y"):
                cv2.destroyWindow("Camera Selection")
                print(f"{camera_index}번 카메라를 사용합니다.")
                return cap, camera_index

            if key == ord("n"):
                cap.release()
                cv2.destroyWindow("Camera Selection")
                break

            if key == ord("q"):
                cap.release()
                cv2.destroyWindow("Camera Selection")
                return None, None

    return None, None


cap, camera_index = open_available_camera(max_camera_index=10)

if cap is None:
    print("사용 가능한 카메라를 선택하지 않았습니다.")
    print("카메라 연결 상태와 권한을 확인하세요.")
    exit()

print(f"최종 선택된 카메라 번호: {camera_index}")
print("웹캠 실시간 감정 분석 시작!")
print("처음 3초 동안 무표정을 유지하세요.")
print("q를 누르면 종료하고 CSV로 저장합니다.")
print("r을 누르면 neutral calibration을 다시 합니다.")


# =========================
# 8. 유틸 함수
# =========================

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value, emotion_name):
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)
    adjusted = max(0, raw_value) * sensitivity

    percent = (adjusted / (adjusted + 1)) * 100
    return clamp(percent)


def get_emotion_color(emotion):
    return COLOR_MAP.get(emotion, (255, 255, 255))


def extract_blendshape_dict(blendshapes):
    data = {}

    for category in blendshapes:
        data[category.category_name] = float(category.score)

    return data


def compute_neutral_profile(neutral_samples):
    all_keys = set()

    for sample in neutral_samples:
        all_keys.update(sample.keys())

    neutral_baseline = {}
    neutral_std = {}

    for key in all_keys:
        values = np.array([sample.get(key, 0.0) for sample in neutral_samples])

        neutral_baseline[key] = float(np.median(values))
        neutral_std[key] = float(np.std(values))

    return neutral_baseline, neutral_std


def make_relative_blendshapes(raw_data, neutral_baseline, neutral_std):
    relative_data = {}

    for key, raw_value in raw_data.items():
        base_value = neutral_baseline.get(key, 0.0)
        std_value = neutral_std.get(key, 0.0)

        deadzone = max(NOISE_FLOOR, std_value * STD_MULTIPLIER)
        delta = raw_value - base_value

        if delta <= deadzone:
            relative_data[key] = 0.0
        else:
            relative_data[key] = float(delta - deadzone)

    return relative_data


def smooth_blendshapes(current_data, previous_data):
    if previous_data is None:
        return current_data.copy()

    smoothed = {}

    for key, current_value in current_data.items():
        previous_value = previous_data.get(key, current_value)

        smoothed[key] = (
            previous_value * SMOOTHING_ALPHA +
            current_value * (1 - SMOOTHING_ALPHA)
        )

    return smoothed


def average_prob_list(prob_list):
    if not prob_list:
        return {}

    all_keys = set()

    for probs in prob_list:
        all_keys.update(probs.keys())

    avg = {}

    for key in all_keys:
        values = [probs.get(key, 0.0) for probs in prob_list]
        avg[key] = sum(values) / len(values)

    return avg


# =========================
# 9. AI-Hub Docker 모델 호출 함수
# =========================

def predict_emotion_with_docker():
    command = [
        "docker",
        "exec",
        DOCKER_CONTAINER,
        "python",
        "emotion.py",
        "--img",
        CROP_PATH_DOCKER,
        "--model_path",
        "model.pth"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print("Docker prediction error:")
        print(result.stderr)
        return "Neutral", 0.0, {}

    output = result.stdout.strip()

    try:
        parsed = ast.literal_eval(output)
        first_result = parsed[0]

        korean_label = first_result.get("label", "중립")
        probs = first_result.get("probs", {})

        english_label = LABEL_MAP.get(korean_label, "Neutral")
        percent = float(probs.get(korean_label, 0.0))

        mapped_probs = {}

        for korean_emotion, value in probs.items():
            english_emotion = LABEL_MAP.get(korean_emotion, korean_emotion)
            mapped_probs[english_emotion] = float(value)

        return english_label, percent, mapped_probs

    except Exception as e:
        print("Result parsing error:", e)
        print("Raw output:", output)
        return "Neutral", 0.0, {}


def choose_aihub_auxiliary_emotion(current_probs):
    if not current_probs:
        return "Neutral", 0.0, {}

    if aihub_baseline_probs is None or not aihub_baseline_probs:
        return "Neutral", 0.0, {}

    delta_probs = {}

    for emotion in AIHUB_AUX_EMOTIONS:
        current_value = current_probs.get(emotion, 0.0)
        baseline_value = aihub_baseline_probs.get(emotion, 0.0)
        delta_probs[emotion] = current_value - baseline_value

    best_label = max(delta_probs, key=delta_probs.get)
    best_delta = delta_probs[best_label]
    best_percent = current_probs.get(best_label, 0.0)

    if best_percent < AIHUB_MIN_CONFIDENCE_TO_USE:
        return "Neutral", best_percent, delta_probs

    if best_delta < AIHUB_DELTA_THRESHOLD:
        return "Neutral", best_percent, delta_probs

    return best_label, best_percent, delta_probs


def run_aihub_prediction_async(frame_idx, is_calibrating_now):
    global is_aihub_predicting
    global aihub_latest_raw_label, aihub_latest_raw_percent
    global aihub_latest_probs, aihub_latest_delta_probs
    global aihub_latest_aux_label, aihub_latest_aux_percent
    global aihub_latest_time

    try:
        raw_label, raw_percent, probs = predict_emotion_with_docker()

        if is_calibrating_now:
            if probs:
                aihub_baseline_samples.append(probs)

            with aihub_lock:
                aihub_latest_raw_label = raw_label
                aihub_latest_raw_percent = raw_percent
                aihub_latest_probs = probs
                aihub_latest_delta_probs = {}
                aihub_latest_aux_label = "Neutral"
                aihub_latest_aux_percent = 0.0
                aihub_latest_time = time.time()

            return

        aux_label, aux_percent, delta_probs = choose_aihub_auxiliary_emotion(probs)

        with aihub_lock:
            aihub_latest_raw_label = raw_label
            aihub_latest_raw_percent = raw_percent
            aihub_latest_probs = probs
            aihub_latest_delta_probs = delta_probs
            aihub_latest_aux_label = aux_label
            aihub_latest_aux_percent = aux_percent
            aihub_latest_time = time.time()

    finally:
        is_aihub_predicting = False


# =========================
# 10. MediaPipe 감정 계산 함수
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
        brow_down_left * 0.25 +
        brow_down_right * 0.25 +
        eye_squint_left * 0.13 +
        eye_squint_right * 0.13 +
        nose_sneer_left * 0.08 +
        nose_sneer_right * 0.08 +
        mouth_press_left * 0.04 +
        mouth_press_right * 0.04
    )

    anger_core = (
        brow_down_left +
        brow_down_right +
        eye_squint_left +
        eye_squint_right +
        mouth_press_left +
        mouth_press_right
    ) / 6

    if smile_avg < 0.08 and eye_wide_avg > 0.20 and anger_core > 0.08:
        anger_raw += eye_wide_avg * 0.15

    if mouth_press_avg > 0.12 and anger_core > 0.08:
        anger_raw += mouth_press_avg * 0.20

    if jaw_open > 0.20:
        anger_raw = anger_raw * (1 - jaw_open * 0.35)

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

    max_emotion_percent = max(
        data["joy"],
        data["sadness"],
        data["anger"],
        data["surprise"]
    )

    data["neutral"] = clamp(100 - max_emotion_percent)
    data["neutral_detected"] = max_emotion_percent < NEUTRAL_THRESHOLD

    return data


def get_primary_emotion(data):
    emotions = {
        "Joy": data["joy"],
        "Sadness": data["sadness"],
        "Anger": data["anger"],
        "Surprise": data["surprise"]
    }

    emotion = max(emotions, key=emotions.get)
    percent = emotions[emotion]

    if percent < NEUTRAL_THRESHOLD:
        return "Neutral", data["neutral"]

    return emotion, percent


def choose_final_emotion_with_aihub(primary_label, primary_percent, data):
    """
    현재 MediaPipe 기반 결과를 우선 사용합니다.
    AI-Hub는 보조 판단기입니다.

    AI-Hub가 최종 감정을 바꿀 수 있는 경우:
    1. MediaPipe 결과가 Neutral인 경우
    2. MediaPipe 결과가 낮은 확신도인 경우
    3. AI-Hub가 baseline 대비 충분히 증가한 비중립 감정을 제안한 경우
    """

    with aihub_lock:
        aux_label = aihub_latest_aux_label
        aux_percent = aihub_latest_aux_percent
        raw_label = aihub_latest_raw_label
        raw_percent = aihub_latest_raw_percent
        probs = dict(aihub_latest_probs)
        delta_probs = dict(aihub_latest_delta_probs)
        last_time = aihub_latest_time

    aihub_age = time.time() - last_time if last_time > 0 else 999.0
    aihub_is_fresh = aihub_age <= AIHUB_PREDICTION_INTERVAL_SECONDS * 2.5

    # 기본값은 MediaPipe 결과
    final_label = primary_label
    final_percent = primary_percent
    final_source = "mediapipe"

    mediapipe_is_ambiguous = (
        primary_label == "Neutral" or
        primary_percent < MEDIAPIPE_STRONG_THRESHOLD
    )

    aihub_can_help = (
        aihub_is_fresh and
        aux_label != "Neutral" and
        aux_percent >= AIHUB_MIN_CONFIDENCE_TO_USE
    )

    if mediapipe_is_ambiguous and aihub_can_help:
        final_label = aux_label
        final_percent = aux_percent
        final_source = "aihub_aux"

    data["mediapipe_label"] = primary_label
    data["mediapipe_percent"] = primary_percent

    data["aihub_raw_label"] = raw_label
    data["aihub_raw_percent"] = raw_percent
    data["aihub_aux_label"] = aux_label
    data["aihub_aux_percent"] = aux_percent
    data["aihub_age"] = aihub_age
    data["final_source"] = final_source

    for emotion_name, value in probs.items():
        data[f"aihub_prob_{emotion_name}"] = value

    for emotion_name, value in delta_probs.items():
        data[f"aihub_delta_{emotion_name}"] = value

    return final_label, final_percent, data


# =========================
# 11. 얼굴 박스 / 화면 표시 함수
# =========================

def get_face_box(face_landmarks, frame_width, frame_height):
    x_values = []
    y_values = []

    for landmark in face_landmarks:
        x_values.append(int(landmark.x * frame_width))
        y_values.append(int(landmark.y * frame_height))

    x_min = max(min(x_values) - 35, 0)
    y_min = max(min(y_values) - 45, 0)
    x_max = min(max(x_values) + 35, frame_width)
    y_max = min(max(y_values) + 35, frame_height)

    return x_min, y_min, x_max, y_max


def draw_emotion_box(frame, box, emotion, percent, source="mediapipe"):
    if box is None:
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

    if source == "aihub_aux":
        text += " +AI"

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
# 12. 실시간 분석 루프
# =========================

results = []
frame_idx = 0

last_box = None
last_emotion = "Neutral"
last_percent = 100
last_source = "mediapipe"

neutral_samples = []
neutral_baseline = None
neutral_std = None
calibration_start_time = time.time()

previous_relative_data = None

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        frame_idx += 1

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        current_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks and detection_result.face_blendshapes:
            face_landmarks = detection_result.face_landmarks[0]
            blendshapes = detection_result.face_blendshapes[0]

            raw_blendshapes = extract_blendshape_dict(blendshapes)
            last_box = get_face_box(face_landmarks, width, height)

            x_min, y_min, x_max, y_max = last_box
            face_crop = frame[y_min:y_max, x_min:x_max]

            should_predict_aihub = (
                not is_aihub_predicting and
                current_time - last_aihub_prediction_time >= AIHUB_PREDICTION_INTERVAL_SECONDS and
                face_crop.size > 0
            )

            if should_predict_aihub:
                cv2.imwrite(CROP_PATH_WINDOWS, face_crop)

                is_aihub_predicting = True
                last_aihub_prediction_time = current_time

                is_calibrating_now = neutral_baseline is None

                thread = threading.Thread(
                    target=run_aihub_prediction_async,
                    args=(frame_idx, is_calibrating_now),
                    daemon=True
                )
                thread.start()

            if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
                if neutral_baseline is None:
                    neutral_samples.append(raw_blendshapes)

                    elapsed = current_time - calibration_start_time

                    if elapsed >= CALIBRATION_SECONDS:
                        if len(neutral_samples) < 5:
                            print("neutral calibration 데이터가 부족합니다.")
                            print("얼굴을 화면에 더 잘 보이게 한 뒤 다시 시도합니다.")

                            neutral_samples = []
                            aihub_baseline_samples.clear()
                            calibration_start_time = time.time()

                        else:
                            neutral_baseline, neutral_std = compute_neutral_profile(neutral_samples)

                            aihub_baseline_probs = average_prob_list(aihub_baseline_samples)

                            print("neutral calibration 완료!")
                            print("수집된 MediaPipe neutral frame 수:", len(neutral_samples))
                            print("수집된 AI-Hub baseline sample 수:", len(aihub_baseline_samples))
                            print("AI-Hub baseline probs:", aihub_baseline_probs)

                            last_emotion = "Neutral"
                            last_percent = 100
                            last_source = "mediapipe"
                            previous_relative_data = None

                else:
                    relative_blendshapes = make_relative_blendshapes(
                        raw_blendshapes,
                        neutral_baseline,
                        neutral_std
                    )

                    relative_blendshapes = smooth_blendshapes(
                        relative_blendshapes,
                        previous_relative_data
                    )

                    previous_relative_data = relative_blendshapes.copy()

                    data = {
                        "frame": frame_idx,
                        "time": current_time
                    }

                    for key, value in relative_blendshapes.items():
                        data[key] = value

                    data = calculate_emotions(data)

                    primary_label, primary_percent = get_primary_emotion(data)

                    final_label, final_percent, data = choose_final_emotion_with_aihub(
                        primary_label,
                        primary_percent,
                        data
                    )

                    last_emotion = final_label
                    last_percent = final_percent
                    last_source = data.get("final_source", "mediapipe")

                    data["dominant_emotion"] = final_label
                    data["dominant_percent"] = final_percent

                    for key, value in raw_blendshapes.items():
                        data[f"raw_{key}"] = value
                        data[f"neutral_base_{key}"] = neutral_baseline.get(key, 0.0)
                        data[f"relative_{key}"] = relative_blendshapes.get(key, 0.0)

                    results.append(data)

        # =========================
        # 화면 표시
        # =========================

        if neutral_baseline is None:
            remaining = max(
                0,
                CALIBRATION_SECONDS - (current_time - calibration_start_time)
            )

            if last_box is not None:
                draw_emotion_box(frame, last_box, "Neutral", 100)

            draw_status_text(
                frame,
                f"Neutral calibration: {remaining:.1f}s",
                40,
                (0, 255, 255)
            )

            draw_status_text(
                frame,
                "Keep a natural neutral face",
                75,
                (0, 255, 255)
            )

        else:
            if last_box is not None:
                draw_emotion_box(
                    frame,
                    last_box,
                    last_emotion,
                    last_percent,
                    last_source
                )

            draw_status_text(
                frame,
                "Analysis mode | q: finish | r: recalibrate",
                height - 30,
                (255, 255, 255)
            )

        if is_aihub_predicting:
            draw_status_text(
                frame,
                "AI-Hub auxiliary analyzing...",
                110,
                (255, 255, 255)
            )

        cv2.imshow("Real-Time Webcam Expression Analysis with AI-Hub Auxiliary", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("q 입력됨. 분석을 종료합니다.")
            break

        if key == ord("r"):
            print("neutral calibration을 다시 시작합니다.")

            neutral_samples = []
            neutral_baseline = None
            neutral_std = None
            calibration_start_time = time.time()
            previous_relative_data = None

            aihub_baseline_samples.clear()
            aihub_baseline_probs = None

            with aihub_lock:
                aihub_latest_raw_label = "Neutral"
                aihub_latest_raw_percent = 0.0
                aihub_latest_probs = {}
                aihub_latest_delta_probs = {}
                aihub_latest_aux_label = "Neutral"
                aihub_latest_aux_percent = 0.0
                aihub_latest_time = 0.0

            last_emotion = "Neutral"
            last_percent = 100
            last_source = "mediapipe"

except KeyboardInterrupt:
    print("\nCtrl+C 입력됨. 분석을 종료하고 CSV 저장 단계로 이동합니다.")

finally:
    cap.release()
    cv2.destroyAllWindows()


# =========================
# 13. CSV 저장
# =========================

if len(results) == 0:
    print("저장할 데이터가 없습니다.")
    print("얼굴이 웹캠에 잘 보이는지 확인하거나, calibration 후 표정을 지어보세요.")

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
        "neutral",
        "mediapipe_label",
        "mediapipe_percent",
        "aihub_raw_label",
        "aihub_raw_percent",
        "aihub_aux_label",
        "aihub_aux_percent",
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