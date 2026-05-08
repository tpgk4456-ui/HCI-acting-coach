import ast
import subprocess
import threading
import time

import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 경로 설정
# =========================

MODEL_PATH = "D:/Seha/HCI/face_landmarker.task"

CROP_PATH_WINDOWS = "D:/Seha/HCI/aihub_input/face.jpg"
CROP_PATH_DOCKER = "/input/face.jpg"

OUTPUT_CSV = "D:/Seha/HCI/user_expression_aihub.csv"

DOCKER_CONTAINER = "facialemotion_runner"


# =========================
# 2. 분석 설정
# =========================

# 시작할 때 무표정 기준을 저장하는 시간
BASELINE_SECONDS = 3.0

# 감정 분석 주기
# 너무 자주 하면 화면이 끊기고, 너무 느리면 반응이 둔함
PREDICTION_INTERVAL_SECONDS = 1.3

# baseline보다 이 정도 이상 증가해야 감정으로 인정
DELTA_THRESHOLD = 18.0

# 이미 표시 중인 감정을 다른 감정으로 바꿀 때 필요한 추가 차이
SWITCH_MARGIN = 12.0

# 모델 confidence가 너무 낮으면 무시
MIN_CONFIDENCE_TO_USE = 35.0


# =========================
# 3. 라벨 / 색상 매핑
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
    "Joy": (0, 255, 255),        # 노랑
    "Anger": (0, 0, 255),        # 빨강
    "Sadness": (255, 0, 0),      # 파랑
    "Surprise": (255, 0, 255),   # 보라
    "Neutral": (255, 255, 255),  # 흰색
    "Anxiety": (0, 165, 255),    # 주황
    "Hurt": (255, 255, 0)        # 하늘색
}

DISPLAY_EMOTIONS = [
    "Joy",
    "Anger",
    "Sadness",
    "Surprise",
    "Anxiety",
    "Hurt"
]


# =========================
# 4. 공유 상태값
# =========================

latest_label = "Neutral"
latest_percent = 0.0
latest_probs = {}

baseline_probs = None
baseline_samples = []
is_calibrating = True
calibration_start_time = None

is_predicting = False
last_prediction_time = 0.0

results = []


# =========================
# 5. 얼굴 박스 함수
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


# =========================
# 6. Docker AI Hub 모델 호출
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


# =========================
# 7. Baseline 관련 함수
# =========================

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


def choose_emotion_with_baseline(current_probs, current_label):
    """
    현재 확률만 보고 감정을 정하지 않고,
    처음 3초 동안 저장한 무표정 baseline과 비교해서 감정을 고름.
    """

    if not current_probs:
        return "Neutral", 0.0, {}

    if baseline_probs is None:
        label = max(current_probs, key=current_probs.get)
        return label, current_probs.get(label, 0.0), {}

    delta_probs = {}

    for emotion in DISPLAY_EMOTIONS:
        current_value = current_probs.get(emotion, 0.0)
        baseline_value = baseline_probs.get(emotion, 0.0)
        delta_probs[emotion] = current_value - baseline_value

    best_label = max(delta_probs, key=delta_probs.get)
    best_delta = delta_probs[best_label]
    best_percent = current_probs.get(best_label, 0.0)

    # 모델 confidence가 너무 낮으면 Neutral
    if best_percent < MIN_CONFIDENCE_TO_USE:
        return "Neutral", current_probs.get("Neutral", 0.0), delta_probs

    # baseline 대비 충분히 증가하지 않았으면 Neutral
    if best_delta < DELTA_THRESHOLD:
        return "Neutral", current_probs.get("Neutral", 0.0), delta_probs

    # 이미 어떤 감정이 표시 중이라면 쉽게 바꾸지 않음
    if current_label != "Neutral" and best_label != current_label:
        current_delta = delta_probs.get(current_label, 0.0)

        if best_delta < current_delta + SWITCH_MARGIN:
            return current_label, current_probs.get(current_label, 0.0), delta_probs

    return best_label, best_percent, delta_probs


# =========================
# 8. 비동기 분석 함수
# =========================

def run_prediction_async(frame_idx):
    global latest_label, latest_percent, latest_probs
    global is_predicting, is_calibrating, baseline_probs

    try:
        raw_label, raw_percent, probs = predict_emotion_with_docker()

        # Calibration 중이면 baseline 샘플만 모음
        if is_calibrating:
            if probs:
                baseline_samples.append(probs)

            row = {
                "frame": frame_idx,
                "mode": "calibration",
                "raw_label": raw_label,
                "raw_percent": raw_percent
            }

            for emotion_name, emotion_percent in probs.items():
                row[emotion_name] = emotion_percent

            results.append(row)
            return

        # Calibration이 끝난 뒤 감정 선택
        display_label, display_percent, delta_probs = choose_emotion_with_baseline(
            probs,
            latest_label
        )

        latest_label = display_label
        latest_percent = display_percent
        latest_probs = probs

        row = {
            "frame": frame_idx,
            "mode": "acting",
            "raw_label": raw_label,
            "raw_percent": raw_percent,
            "displayed_label": display_label,
            "displayed_percent": display_percent
        }

        for emotion_name, emotion_percent in probs.items():
            row[emotion_name] = emotion_percent

        for emotion_name, delta_value in delta_probs.items():
            row[f"{emotion_name}_delta"] = delta_value

        results.append(row)

    finally:
        is_predicting = False


# =========================
# 9. 화면 표시 함수
# =========================

def draw_label(frame, box, label, percent):
    x_min, y_min, x_max, y_max = box
    color = COLOR_MAP.get(label, (255, 255, 255))

    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        color,
        3
    )

    text = f"{label} {percent:.0f}%"

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
        color,
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
        0.7,
        color,
        2
    )


# =========================
# 10. MediaPipe 설정
# =========================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# 11. 카메라 열기
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("AI Hub 감정 인식 웹캠 분석 시작")
print("처음 3초 동안 무표정 calibration을 진행합니다.")
print("q를 누르면 종료하고 CSV를 저장합니다.")


# =========================
# 12. 실시간 분석 루프
# =========================

frame_idx = 0
last_box = None
calibration_start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    frame_idx += 1

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    current_time = time.time()
    elapsed_calibration = current_time - calibration_start_time

    # Calibration 종료 처리
    if is_calibrating and elapsed_calibration >= BASELINE_SECONDS:
        baseline_probs = average_prob_list(baseline_samples)
        is_calibrating = False

        latest_label = "Neutral"
        latest_percent = 0.0

        print("Calibration complete.")
        print("Baseline probs:", baseline_probs)

    # MediaPipe 얼굴 추적
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        box = get_face_box(face_landmarks, width, height)
        last_box = box

        x_min, y_min, x_max, y_max = box
        face_crop = frame[y_min:y_max, x_min:x_max]

        should_predict = (
            not is_predicting
            and current_time - last_prediction_time >= PREDICTION_INTERVAL_SECONDS
            and face_crop.size > 0
        )

        if should_predict:
            cv2.imwrite(CROP_PATH_WINDOWS, face_crop)

            is_predicting = True
            last_prediction_time = current_time

            thread = threading.Thread(
                target=run_prediction_async,
                args=(frame_idx,),
                daemon=True
            )
            thread.start()

    # 얼굴 박스 그리기
    if last_box is not None:
        if is_calibrating:
            draw_label(frame, last_box, "Neutral", 0.0)
        else:
            draw_label(frame, last_box, latest_label, latest_percent)

    # 상태 문구 표시
    if is_calibrating:
        remaining = max(BASELINE_SECONDS - elapsed_calibration, 0)
        draw_status_text(
            frame,
            f"Calibrating neutral face... {remaining:.1f}s",
            40,
            (0, 255, 255)
        )
        draw_status_text(
            frame,
            "Keep a neutral expression.",
            75,
            (0, 255, 255)
        )
    else:
        draw_status_text(
            frame,
            "Start acting!",
            40,
            (255, 255, 255)
        )

    if is_predicting:
        draw_status_text(
            frame,
            "Analyzing...",
            110,
            (255, 255, 255)
        )

    draw_status_text(
        frame,
        "Press q to finish",
        height - 30,
        (255, 255, 255)
    )

    cv2.imshow("AI Hub Emotion Recognition with Neutral Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 13. 종료 후 CSV 저장
# =========================

cap.release()
cv2.destroyAllWindows()

if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료:", OUTPUT_CSV)
    print(df.tail())
else:
    print("저장할 결과가 없습니다.")