import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

model_path = "face_landmarker.task"
video_path = "actor_video.mp4"
output_csv = "video_expression_neutral.csv"


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
NEUTRAL_WINDOW_SECONDS = 1.0

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
# 4. 유틸 함수
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
# 5. neutral 자동 선택 함수
# =========================

def calculate_neutral_score(raw_data, previous_raw_data=None):
    """
    배우 영상은 처음 3초 무표정을 요구할 수 없으므로,
    표정 신호와 프레임 간 움직임이 가장 작은 구간을 neutral 후보로 고릅니다.
    """

    mouth_open = raw_data.get("jawOpen", 0.0)
    mouth_corner = (
        raw_data.get("mouthSmileLeft", 0.0) +
        raw_data.get("mouthSmileRight", 0.0) +
        raw_data.get("mouthFrownLeft", 0.0) +
        raw_data.get("mouthFrownRight", 0.0)
    ) / 4
    mouth_press = (
        raw_data.get("mouthPressLeft", 0.0) +
        raw_data.get("mouthPressRight", 0.0)
    ) / 2
    brow_movement = (
        raw_data.get("browInnerUp", 0.0) +
        raw_data.get("browDownLeft", 0.0) +
        raw_data.get("browDownRight", 0.0) +
        raw_data.get("browOuterUpLeft", 0.0) +
        raw_data.get("browOuterUpRight", 0.0)
    ) / 5
    eye_movement = (
        raw_data.get("eyeWideLeft", 0.0) +
        raw_data.get("eyeWideRight", 0.0) +
        raw_data.get("eyeSquintLeft", 0.0) +
        raw_data.get("eyeSquintRight", 0.0)
    ) / 4
    nose_movement = (
        raw_data.get("noseSneerLeft", 0.0) +
        raw_data.get("noseSneerRight", 0.0)
    ) / 2

    motion_score = 0.0

    if previous_raw_data is not None:
        motion_keys = [
            "jawOpen",
            "mouthSmileLeft",
            "mouthSmileRight",
            "mouthFrownLeft",
            "mouthFrownRight",
            "browInnerUp",
            "browDownLeft",
            "browDownRight",
            "browOuterUpLeft",
            "browOuterUpRight",
            "eyeWideLeft",
            "eyeWideRight",
            "eyeSquintLeft",
            "eyeSquintRight"
        ]

        motion_values = []

        for key in motion_keys:
            motion_values.append(abs(raw_data.get(key, 0.0) - previous_raw_data.get(key, 0.0)))

        motion_score = float(np.mean(motion_values))

    return (
        mouth_open * 1.4 +
        mouth_corner * 1.2 +
        mouth_press * 0.8 +
        brow_movement * 1.1 +
        eye_movement * 0.9 +
        nose_movement * 0.7 +
        motion_score * 2.0
    )


def compute_neutral_profile(neutral_samples):
    all_keys = set()

    for sample in neutral_samples:
        all_keys.update(sample.keys())

    neutral_baseline = {}
    neutral_std = {}

    for key in all_keys:
        values = np.array([sample.get(key, 0.0) for sample in neutral_samples])

        # 평균보다 median이 순간 표정과 인식 튐에 강함
        neutral_baseline[key] = float(np.median(values))
        neutral_std[key] = float(np.std(values))

    return neutral_baseline, neutral_std


def select_neutral_window(samples, fps):
    if not samples:
        return [], None

    window_size = max(1, int((fps / ANALYZE_EVERY_N_FRAMES) * NEUTRAL_WINDOW_SECONDS))

    if len(samples) <= window_size:
        return samples, {
            "start_frame": samples[0]["frame"],
            "end_frame": samples[-1]["frame"],
            "score": float(np.mean([sample["neutral_score"] for sample in samples]))
        }

    best_start_index = 0
    best_score = None

    for start_index in range(0, len(samples) - window_size + 1):
        window = samples[start_index:start_index + window_size]
        window_score = float(np.mean([sample["neutral_score"] for sample in window]))

        if best_score is None or window_score < best_score:
            best_score = window_score
            best_start_index = start_index

    selected = samples[best_start_index:best_start_index + window_size]

    return selected, {
        "start_frame": selected[0]["frame"],
        "end_frame": selected[-1]["frame"],
        "score": best_score
    }


def collect_video_samples(video_capture, fps, total_frames):
    samples = []
    frame_idx = 0
    previous_raw_data = None

    print("neutral 후보 구간 탐색 중...")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_idx += 1

        if frame_idx % ANALYZE_EVERY_N_FRAMES != 0:
            continue

        face_landmarks, blendshapes = detect_frame(frame)

        if face_landmarks is None or blendshapes is None:
            continue

        raw_blendshapes = extract_blendshape_dict(blendshapes)
        neutral_score = calculate_neutral_score(raw_blendshapes, previous_raw_data)
        previous_raw_data = raw_blendshapes

        samples.append({
            "frame": frame_idx,
            "time": frame_idx / fps,
            "neutral_score": neutral_score,
            "raw_blendshapes": raw_blendshapes
        })

        if len(samples) % 100 == 0:
            progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"neutral 후보 탐색 진행률: {progress:.1f}%")

    return samples


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

    # Joy: 입꼬리 상승 + 약간의 눈웃음
    joy_raw = (
        mouth_smile_left * 0.45 +
        mouth_smile_right * 0.45 +
        eye_squint_left * 0.05 +
        eye_squint_right * 0.05
    )

    # Sadness: 입꼬리 내려감 + 안쪽 눈썹 상승 + 입술 누름 약간
    sadness_raw = (
        mouth_frown_left * 0.30 +
        mouth_frown_right * 0.30 +
        brow_inner_up * 0.35 +
        mouth_press_left * 0.025 +
        mouth_press_right * 0.025
    )

    # Anger: 눈썹 내려감 + 눈 찡그림 + 코 찡그림 + 입술 누름
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

    # Surprise: 입 벌림 + 눈썹 상승 + 눈 커짐
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


def get_dominant_emotion(data):
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


def draw_emotion_box(frame, box, emotion, percent):
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


# =========================
# 7. 영상 열기 / neutral 기준 계산
# =========================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다. video_path를 확인하세요.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30

delay = int(1000 / fps)

print("영상 neutral 기준 자동 탐색 시작!")
print("FPS:", fps)
print("총 프레임 수:", total_frames)

all_samples = collect_video_samples(cap, fps, total_frames)
neutral_window, neutral_info = select_neutral_window(all_samples, fps)

if not neutral_window:
    print("neutral 기준을 만들 수 없습니다. 영상에서 얼굴이 잘 보이는지 확인하세요.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

neutral_samples = [sample["raw_blendshapes"] for sample in neutral_window]
neutral_baseline, neutral_std = compute_neutral_profile(neutral_samples)

print("neutral 기준 자동 선택 완료!")
print("선택 구간:", neutral_info["start_frame"], "~", neutral_info["end_frame"], "frame")
print("선택 구간 시간:", f"{neutral_info['start_frame'] / fps:.2f}s", "~", f"{neutral_info['end_frame'] / fps:.2f}s")
print("neutral score:", f"{neutral_info['score']:.4f}")


# =========================
# 8. 영상 분석 루프
# =========================

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

results = []
frame_idx = 0

last_box = None
last_emotion = "Neutral"
last_percent = 100
previous_relative_data = None

print("영상 neutral 기준 보정 분석 시작!")
print("q를 누르면 종료하고 CSV로 저장합니다.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 분석 완료.")
        break

    frame_idx += 1
    height, width, _ = frame.shape

    if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
        face_landmarks, blendshapes = detect_frame(frame)

        if face_landmarks is not None and blendshapes is not None:
            raw_blendshapes = extract_blendshape_dict(blendshapes)
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
                "time": frame_idx / fps,
                "neutral_start_frame": neutral_info["start_frame"],
                "neutral_end_frame": neutral_info["end_frame"],
                "neutral_score": neutral_info["score"]
            }

            for key, value in relative_blendshapes.items():
                data[key] = value

            data = calculate_emotions(data)
            emotion, percent = get_dominant_emotion(data)

            last_box = get_face_box(face_landmarks, width, height)
            last_emotion = emotion
            last_percent = percent

            data["dominant_emotion"] = emotion
            data["dominant_percent"] = percent

            for key, value in raw_blendshapes.items():
                data[f"raw_{key}"] = value
                data[f"neutral_base_{key}"] = neutral_baseline.get(key, 0.0)
                data[f"relative_{key}"] = relative_blendshapes.get(key, 0.0)

            results.append(data)

    if last_box is not None:
        draw_emotion_box(frame, last_box, last_emotion, last_percent)

    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0

    cv2.putText(
        frame,
        f"Progress: {progress:.1f}% | Neutral: {neutral_info['start_frame']}-{neutral_info['end_frame']}",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Video Expression Analysis with Neutral Baseline", frame)

    key = cv2.waitKey(delay) & 0xFF

    if key == ord("q"):
        print("q 입력됨. 분석을 종료합니다.")
        break


# =========================
# 9. CSV 저장
# =========================

cap.release()
cv2.destroyAllWindows()

if len(results) == 0:
    print("저장할 데이터가 없습니다. 영상에서 얼굴이 잘 보이는지 확인하세요.")

else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))

    emotion_cols = ["joy", "sadness", "anger", "surprise", "neutral"]

    print("\n영상 평균 감정 표현 강도:")
    print(df[emotion_cols].mean())

    print("\n감정 분류 개수:")
    print(df["dominant_emotion"].value_counts())
