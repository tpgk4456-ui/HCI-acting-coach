import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

model_path = "/Users/parkseoyeon/Junior/HCI/HCI-acting-coach/face_landmarker.task"
output_csv = "user_expression_custom.csv"


# =========================
# 2. 감정 민감도 / 설정값
# =========================

SENSITIVITY = {
    "joy": 4.0,
    "sadness": 6.0,
    "anger": 4.0,
    "surprise": 4.5
}

CALIBRATION_SECONDS = 3.0
ANALYZE_EVERY_N_FRAMES = 3

# neutral 기준에서 이 정도 차이는 표정 변화가 아니라 노이즈로 처리
NOISE_FLOOR = 0.015
STD_MULTIPLIER = 2.0

# 가장 강한 감정 점수가 이 값보다 낮으면 neutral로 판단
NEUTRAL_THRESHOLD = 18

# 값이 너무 덜컥덜컥 바뀌지 않도록 smoothing
SMOOTHING_ALPHA = 0.65


# =========================
# 3. MediaPipe Face Landmarker 설정
# =========================

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# 4. 웹캠 자동 선택
# =========================

def open_available_camera(max_camera_index=5):
    for camera_index in range(max_camera_index + 1):
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                print(f"{camera_index}번 카메라를 사용합니다.")
                return cap

            cap.release()

    return None


cap = open_available_camera(max_camera_index=5)

if cap is None:
    print("사용 가능한 카메라를 찾을 수 없습니다.")
    print("카메라 연결 상태와 권한을 확인하세요.")
    print("Mac: 시스템 설정 → 개인정보 보호 및 보안 → 카메라")
    print("Windows: 설정 → 개인 정보 및 보안 → 카메라")
    exit()

print("웹캠 실시간 감정 분석 시작!")
print("처음 3초 동안 무표정을 유지하세요.")
print("q를 누르면 종료하고 CSV로 저장합니다.")
print("r을 누르면 neutral calibration을 다시 합니다.")


# =========================
# 5. 유틸 함수
# =========================

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value, emotion_name):
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)
    adjusted = max(0, raw_value) * sensitivity

    percent = (adjusted / (adjusted + 1)) * 100
    return clamp(percent)


def get_emotion_color(emotion):
    # OpenCV는 RGB가 아니라 BGR 순서
    colors = {
        "Joy": (0, 255, 255),        # 노란색
        "Sadness": (255, 0, 0),      # 파란색
        "Anger": (0, 0, 255),        # 빨간색
        "Surprise": (255, 0, 255),   # 보라색
        "Neutral": (255, 255, 255)   # 흰색
    }

    return colors.get(emotion, (255, 255, 255))


def extract_blendshape_dict(blendshapes):
    data = {}

    for category in blendshapes:
        data[category.category_name] = float(category.score)

    return data


def compute_neutral_profile(neutral_samples):
    """
    3초 동안 모은 blendshape 값으로
    사용자의 평소 무표정 baseline과 흔들림 정도를 계산합니다.
    """

    all_keys = set()

    for sample in neutral_samples:
        all_keys.update(sample.keys())

    neutral_baseline = {}
    neutral_std = {}

    for key in all_keys:
        values = np.array([sample.get(key, 0.0) for sample in neutral_samples])

        # 평균보다 median이 순간 노이즈에 강함
        neutral_baseline[key] = float(np.median(values))
        neutral_std[key] = float(np.std(values))

    return neutral_baseline, neutral_std


def make_relative_blendshapes(raw_data, neutral_baseline, neutral_std):
    """
    절대 blendshape 값이 아니라,
    사용자의 평소 무표정 대비 얼마나 증가했는지 계산합니다.
    """

    relative_data = {}

    for key, raw_value in raw_data.items():
        base_value = neutral_baseline.get(key, 0.0)
        std_value = neutral_std.get(key, 0.0)

        deadzone = max(NOISE_FLOOR, std_value * STD_MULTIPLIER)

        delta = raw_value - base_value

        # neutral과 거의 같은 값이면 0으로 처리
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


# =========================
# 6. 감정 계산 함수
# =========================

def calculate_emotions(data):
    """
    여기서 data는 raw blendshape가 아니라
    neutral 기준으로 보정된 relative blendshape입니다.
    """

    # 입 관련
    mouth_smile_left = data.get("mouthSmileLeft", 0)
    mouth_smile_right = data.get("mouthSmileRight", 0)

    mouth_frown_left = data.get("mouthFrownLeft", 0)
    mouth_frown_right = data.get("mouthFrownRight", 0)

    mouth_press_left = data.get("mouthPressLeft", 0)
    mouth_press_right = data.get("mouthPressRight", 0)

    # 눈썹 관련
    brow_inner_up = data.get("browInnerUp", 0)
    brow_down_left = data.get("browDownLeft", 0)
    brow_down_right = data.get("browDownRight", 0)

    brow_outer_up_left = data.get("browOuterUpLeft", 0)
    brow_outer_up_right = data.get("browOuterUpRight", 0)

    # 눈 관련
    eye_squint_left = data.get("eyeSquintLeft", 0)
    eye_squint_right = data.get("eyeSquintRight", 0)

    eye_wide_left = data.get("eyeWideLeft", 0)
    eye_wide_right = data.get("eyeWideRight", 0)

    # 코 관련
    nose_sneer_left = data.get("noseSneerLeft", 0)
    nose_sneer_right = data.get("noseSneerRight", 0)

    # 턱 / 입 벌림
    jaw_open = data.get("jawOpen", 0)

    # 평균값
    smile_avg = (mouth_smile_left + mouth_smile_right) / 2
    eye_wide_avg = (eye_wide_left + eye_wide_right) / 2
    mouth_press_avg = (mouth_press_left + mouth_press_right) / 2
    brow_outer_up_avg = (brow_outer_up_left + brow_outer_up_right) / 2
    brow_up_avg = (brow_inner_up + brow_outer_up_avg) / 2

    # =========================
    # Joy
    # 입꼬리 상승 + 약간의 눈웃음
    # =========================
    joy_raw = (
        mouth_smile_left * 0.45 +
        mouth_smile_right * 0.45 +
        eye_squint_left * 0.05 +
        eye_squint_right * 0.05
    )

    # =========================
    # Sadness
    # 입꼬리 내려감 + 안쪽 눈썹 상승 + 입술 누름 약간
    # =========================
    sadness_raw = (
        mouth_frown_left * 0.30 +
        mouth_frown_right * 0.30 +
        brow_inner_up * 0.35 +
        mouth_press_left * 0.025 +
        mouth_press_right * 0.025
    )

    # =========================
    # Anger
    # 눈썹 내려감 + 눈 찡그림 + 코 찡그림 + 입술 누름
    # =========================
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

    # 눈만 커진 경우를 anger로 과하게 보정하지 않도록 조건 강화
    if smile_avg < 0.08 and eye_wide_avg > 0.20 and anger_core > 0.08:
        anger_raw += eye_wide_avg * 0.15

    # 입술을 누르는 값도 anger core가 있을 때만 보정
    if mouth_press_avg > 0.12 and anger_core > 0.08:
        anger_raw += mouth_press_avg * 0.20

    # 입이 많이 벌어져 있으면 anger 점수 감소
    if jaw_open > 0.20:
        anger_raw = anger_raw * (1 - jaw_open * 0.35)

    # =========================
    # Surprise
    # 입 벌림 + 눈썹 상승 + 눈 커짐
    # =========================
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

    # 분노 신호가 강하면 surprise 감점
    surprise_raw *= (1 - anger_signal * 0.5)

    # 입만 벌린 경우 surprise 과대 인식 방지
    if jaw_open > 0.4 and brow_inner_up < 0.15 and eye_wide_avg < 0.15:
        surprise_raw *= 0.4

    # 눈은 커졌지만 입을 크게 벌리지 않았다면 surprise 감소
    if smile_avg < 0.15 and eye_wide_avg > 0.18 and jaw_open < 0.25:
        surprise_raw *= 0.5

    # CSV 저장용 raw emotion score
    data["joy_raw"] = joy_raw
    data["sadness_raw"] = sadness_raw
    data["anger_raw"] = anger_raw
    data["surprise_raw"] = surprise_raw

    # 화면 / CSV 표시용 percent
    data["joy"] = raw_to_percent(joy_raw, "joy")
    data["sadness"] = raw_to_percent(sadness_raw, "sadness")
    data["anger"] = raw_to_percent(anger_raw, "anger")
    data["surprise"] = raw_to_percent(surprise_raw, "surprise")

    # =========================
    # Neutral column 추가
    # =========================
    max_emotion_percent = max(
        data["joy"],
        data["sadness"],
        data["anger"],
        data["surprise"]
    )

    # 표정 변화가 거의 없을수록 neutral이 높음
    data["neutral"] = clamp(100 - max_emotion_percent)

    # 실제 neutral로 판단했는지 여부
    data["neutral_detected"] = max_emotion_percent < NEUTRAL_THRESHOLD

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


def get_dominant_emotion(data):
    emotions = {
        "Joy": data["joy"],
        "Sadness": data["sadness"],
        "Anger": data["anger"],
        "Surprise": data["surprise"]
    }

    emotion = max(emotions, key=emotions.get)
    percent = emotions[emotion]

    # 감정 강도가 낮으면 neutral로 판단
    if percent < NEUTRAL_THRESHOLD:
        return "Neutral", data["neutral"]

    return emotion, percent


def draw_emotion_box(frame, box, emotion, percent):
    if box is None:
        return

    x_min, y_min, x_max, y_max = box
    box_color = get_emotion_color(emotion)

    # 얼굴 추적 박스
    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        box_color,
        3
    )

    # 텍스트
    text = f"{emotion} {percent:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    text_x = x_min
    text_y = max(y_min - 12, text_height + 12)

    bg_x1 = text_x
    bg_y1 = text_y - text_height - 10
    bg_x2 = text_x + text_width + 12
    bg_y2 = text_y + 6

    cv2.rectangle(
        frame,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        box_color,
        -1
    )

    text_color = (0, 0, 0)

    cv2.putText(
        frame,
        text,
        (text_x + 6, text_y),
        font,
        font_scale,
        text_color,
        thickness
    )


# =========================
# 7. 웹캠 실시간 분석 루프
# =========================

results = []
frame_idx = 0

last_box = None
last_emotion = "Neutral"
last_percent = 100

neutral_samples = []
neutral_baseline = None
neutral_std = None
calibration_start_time = time.time()

previous_relative_data = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("웹캠 프레임을 읽을 수 없습니다.")
        break

    frame_idx += 1

    # 거울처럼 보이게 좌우 반전
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    current_time = time.time()

    if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
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

            # =========================
            # neutral calibration 단계
            # =========================
            if neutral_baseline is None:
                neutral_samples.append(raw_blendshapes)

                elapsed = current_time - calibration_start_time

                if elapsed >= CALIBRATION_SECONDS:
                    if len(neutral_samples) < 5:
                        print("neutral calibration 데이터가 부족합니다.")
                        print("얼굴을 화면에 더 잘 보이게 한 뒤 다시 시도합니다.")

                        neutral_samples = []
                        calibration_start_time = time.time()

                    else:
                        neutral_baseline, neutral_std = compute_neutral_profile(neutral_samples)

                        print("neutral calibration 완료!")
                        print("수집된 neutral frame 수:", len(neutral_samples))

                        last_emotion = "Neutral"
                        last_percent = 100
                        previous_relative_data = None

            # =========================
            # 실제 감정 분석 단계
            # =========================
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
                    "frame": frame_idx
                }

                # 기존 blendshape 이름에는 relative 값을 저장
                # 예: mouthSmileLeft = neutral 대비 증가량
                for key, value in relative_blendshapes.items():
                    data[key] = value

                data = calculate_emotions(data)

                emotion, percent = get_dominant_emotion(data)

                last_emotion = emotion
                last_percent = percent

                data["dominant_emotion"] = emotion
                data["dominant_percent"] = percent

                # CSV 확인용으로 raw / neutral baseline / relative 모두 저장
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

        cv2.putText(
            frame,
            f"Neutral calibration: {remaining:.1f}s",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Keep a natural neutral face",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    else:
        if last_box is not None:
            draw_emotion_box(frame, last_box, last_emotion, last_percent)

        cv2.putText(
            frame,
            "Analysis mode | q: finish | r: recalibrate",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    cv2.imshow("Real-Time Webcam Expression Analysis", frame)

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

        last_emotion = "Neutral"
        last_percent = 100


# =========================
# 8. CSV 저장
# =========================

cap.release()
cv2.destroyAllWindows()

if len(results) == 0:
    print("저장할 데이터가 없습니다.")
    print("얼굴이 웹캠에 잘 보이는지 확인하거나, calibration 후 표정을 지어보세요.")

else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))

    emotion_cols = ["joy", "sadness", "anger", "surprise", "neutral"]

    print("\n사용자 평균 감정 표현 강도:")
    print(df[emotion_cols].mean())

    print("\n감정 분류 개수:")
    print(df["dominant_emotion"].value_counts())