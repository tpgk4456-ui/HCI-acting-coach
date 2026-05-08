import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

model_path = "face_landmarker.task"
output_csv = "user_expression_custom.csv"


# =========================
# 2. 감정 민감도 설정
# =========================

SENSITIVITY = {
    "joy": 4.0,
    "sadness": 6.0,
    "anger": 4.0,
    "surprise": 4.5
}


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
# 4. 웹캠 열기
# =========================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("1번 카메라를 열 수 없습니다.")
    exit()

if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")
    exit()

print("웹캠 실시간 감정 분석 시작!")
print("q를 누르면 종료하고 CSV로 저장합니다.")


# =========================
# 5. 유틸 함수
# =========================

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value, emotion_name):
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)
    adjusted = raw_value * sensitivity

    # 값이 너무 쉽게 100%가 되지 않도록 완만하게 변환
    percent = (adjusted / (adjusted + 1)) * 100

    return clamp(percent)


def get_emotion_color(emotion):
    # OpenCV는 RGB가 아니라 BGR 순서
    colors = {
        "Joy": (0, 255, 255),       # 노란색
        "Sadness": (255, 0, 0),     # 파란색
        "Anger": (0, 0, 255),       # 빨간색
        "Surprise": (255, 0, 255),  # 보라색
        "Neutral": (255, 255, 255)  # 흰색
    }

    return colors.get(emotion, (255, 255, 255))


# =========================
# 6. 감정 계산 함수
# =========================

def calculate_emotions(data):
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

    #surprise 관련 데이터 추가함
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
    # 추가 조건:
    # 입꼬리가 많이 안 올라가 있고, 눈이 크게 떠져 있으면 Anger 보정
    # =========================
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

    # 입꼬리가 별로 안 올라가 있고 눈이 커져 있으면 분노로 보정
    if smile_avg < 0.15 and eye_wide_avg > 0.18:
        anger_raw += eye_wide_avg * 0.35

    # 입을 앙 다문 상태면 분노 보정
    if mouth_press_avg > 0.15:
        anger_raw += mouth_press_avg * 0.25

    # 입이 너무 많이 벌어져 있으면 Anger 점수를 조금 낮춤
    # 단, 눈이 크게 떠져 있고 입꼬리가 안 올라간 경우는 완전히 깎지 않음
    if not (smile_avg < 0.15 and eye_wide_avg > 0.18):
        anger_raw = anger_raw * (1 - jaw_open * 0.4)

    # =========================
    # Surprise
    # 입 벌림 + 눈썹 상승 + 눈 커짐
    # =========================
    '''surprise_raw = (
        jaw_open * 0.35 +
        brow_inner_up * 0.25 +
        eye_wide_left * 0.20 +
        eye_wide_right * 0.20
    )'''
    
    anger_signal = (
        brow_down_left + brow_down_right +
        eye_squint_left + eye_squint_right +
        nose_sneer_left + nose_sneer_right +
        mouth_press_left + mouth_press_right
    ) / 8

    # Surprise는 눈썹 전체 상승 + 눈 커짐 + 입 벌림이 같이 있어야 높게
    surprise_raw = (
        jaw_open * 0.30 +
        brow_up_avg * 0.30 +
        eye_wide_left * 0.20 +
        eye_wide_right * 0.20
    )

    # 분노 신호가 강하면 Surprise 감점
    surprise_raw *= (1 - anger_signal * 0.5)

    # 입만 벌린 경우 Surprise 과대 인식 방지
    if jaw_open > 0.4 and brow_inner_up < 0.15 and eye_wide_avg < 0.15:
        surprise_raw *= 0.4

    # 입꼬리가 안 올라가 있고, 눈은 커졌지만 입을 크게 벌리지 않았다면
    # Surprise보다는 Anger 쪽으로 보도록 Surprise를 낮춤
    if smile_avg < 0.15 and eye_wide_avg > 0.18 and jaw_open < 0.25:
        surprise_raw *= 0.5

    # CSV 저장용 원본값
    data["joy_raw"] = joy_raw
    data["sadness_raw"] = sadness_raw
    data["anger_raw"] = anger_raw
    data["surprise_raw"] = surprise_raw

    # 화면 표시용 퍼센트
    data["joy"] = raw_to_percent(joy_raw, "joy")
    data["sadness"] = raw_to_percent(sadness_raw, "sadness")
    data["anger"] = raw_to_percent(anger_raw, "anger")
    data["surprise"] = raw_to_percent(surprise_raw, "surprise")

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

    # 너무 낮으면 무표정으로 처리
    if percent < 15:
        return "Neutral", 0

    return emotion, percent


def draw_emotion_box(frame, box, emotion, percent):
    # Neutral이면 박스 표시 안 함
    if emotion == "Neutral":
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
# 7. 웹캠 실시간 분석 루프
# =========================

results = []
frame_idx = 0

last_box = None
last_emotion = "Neutral"
last_percent = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("웹캠 프레임을 읽을 수 없습니다.")
        break

    frame_idx += 1

    # 거울처럼 보이게 좌우 반전
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # 3프레임마다 분석
    # 더 민감하게 보고 싶으면 2 또는 1로 바꿔도 됨
    if frame_idx % 3 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks and detection_result.face_blendshapes:
            face_landmarks = detection_result.face_landmarks[0]
            blendshapes = detection_result.face_blendshapes[0]

            data = {
                "frame": frame_idx
            }

            for category in blendshapes:
                data[category.category_name] = category.score

            data = calculate_emotions(data)

            emotion, percent = get_dominant_emotion(data)

            last_box = get_face_box(face_landmarks, width, height)
            last_emotion = emotion
            last_percent = percent

            data["dominant_emotion"] = emotion
            data["dominant_percent"] = percent

            results.append(data)

    # 얼굴 박스와 감정 표시
    if last_box is not None:
        draw_emotion_box(frame, last_box, last_emotion, last_percent)

    cv2.putText(
        frame,
        "Press q to finish",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Real-Time Webcam Expression Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("q 입력됨. 분석을 종료합니다.")
        break


# =========================
# 8. CSV 저장
# =========================

cap.release()
cv2.destroyAllWindows()

if len(results) == 0:
    print("저장할 데이터가 없습니다. 얼굴이 웹캠에 잘 보이는지 확인하세요.")
else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))

    emotion_cols = ["joy", "sadness", "anger", "surprise"]
    print("\n사용자 평균 감정 표현 강도:")
    print(df[emotion_cols].mean())