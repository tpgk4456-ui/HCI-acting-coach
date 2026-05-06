import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

model_path = "D:/Seha/HCI/face_landmarker.task"
output_csv = "D:/Seha/HCI/user_expression.csv"


# =========================
# 2. 감정 민감도 설정
# =========================
# 값이 클수록 퍼센트가 더 잘 올라감
# 화남이 너무 쉽게 100%가 뜨면 anger 값을 더 낮추면 됨

SENSITIVITY = {
    "joy": 4.0,
    "sadness": 7.0,
    "anger": 1.0,
    "surprise": 3.5
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
# 기본 카메라가 0번, 로지텍 외장 카메라가 1번일 수 있음

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("0번 카메라를 열 수 없습니다. 1번 카메라를 시도합니다.")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")
    exit()

print("웹캠 분석 시작!")
print("q를 누르면 종료하고 CSV로 저장합니다.")


# =========================
# 5. 유틸 함수
# =========================

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value, emotion_name):
    """
    원본 blendshape 값은 작게 나오거나 특정 표정에서 갑자기 튈 수 있음.
    그래서 단순히 raw * sensitivity * 100을 하지 않고,
    완만하게 증가하는 방식으로 퍼센트를 계산함.
    """
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)

    adjusted = raw_value * sensitivity

    # 완만한 증가 함수: 값이 커져도 바로 100%가 되지 않음
    percent = (adjusted / (adjusted + 1)) * 100

    return clamp(percent)


def get_emotion_color(emotion):
    """
    OpenCV는 RGB가 아니라 BGR 순서임.
    """
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
    mouth_smile_left = data.get("mouthSmileLeft", 0)
    mouth_smile_right = data.get("mouthSmileRight", 0)

    mouth_frown_left = data.get("mouthFrownLeft", 0)
    mouth_frown_right = data.get("mouthFrownRight", 0)

    brow_inner_up = data.get("browInnerUp", 0)
    brow_down_left = data.get("browDownLeft", 0)
    brow_down_right = data.get("browDownRight", 0)

    eye_squint_left = data.get("eyeSquintLeft", 0)
    eye_squint_right = data.get("eyeSquintRight", 0)

    jaw_open = data.get("jawOpen", 0)

    # 원본 표정 점수
    joy_raw = (mouth_smile_left + mouth_smile_right) / 2
    sadness_raw = (mouth_frown_left + mouth_frown_right + brow_inner_up) / 3
    anger_raw = (
        brow_down_left +
        brow_down_right +
        eye_squint_left +
        eye_squint_right
    ) / 4
    surprise_raw = (jaw_open + brow_inner_up) / 2

    # CSV에는 원본값도 저장
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


# =========================
# 7. 웹캠 분석 루프
# =========================

results = []
frame_idx = 0

last_box = None
last_emotion = "Neutral"
last_percent = 0
last_label = "Neutral 0%"

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
            last_label = f"{emotion} {percent:.0f}%"

            results.append(data)

    # =========================
    # 8. 얼굴 박스와 감정 표시
    # =========================

    if last_box is not None:
        x_min, y_min, x_max, y_max = last_box

        box_color = get_emotion_color(last_emotion)

        # 얼굴 추적 박스
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            box_color,
            3
        )

        # 텍스트 배경 박스
        text = last_label
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

    cv2.putText(
        frame,
        "Press q to finish",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Tracking Expression Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("q 입력됨. 분석을 종료합니다.")
        break


# =========================
# 9. CSV 저장
# =========================

cap.release()
cv2.destroyAllWindows()

if len(results) == 0:
    print("저장할 데이터가 없습니다. 얼굴이 웹캠에 잘 보이는지 확인하세요.")
else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("user_expression.csv 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))

    emotion_cols = ["joy", "sadness", "anger", "surprise"]
    print("\n사용자 평균 감정 표현 강도:")
    print(df[emotion_cols].mean())