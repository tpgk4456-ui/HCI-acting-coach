import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 파일 경로 설정
# =========================

video_path = "D:/Seha/HCI/actor_video.mp4"
model_path = "D:/Seha/HCI/face_landmarker.task"
output_csv = "D:/Seha/HCI/actor_expression.csv"


# =========================
# 2. MediaPipe Face Landmarker 설정
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
# 3. OpenCV로 mp4 영상 열기
# =========================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다. video_path를 확인하세요.")
    exit()

print("영상 열기 성공. 분석을 시작합니다.")


# =========================
# 4. 영상 프레임 분석
# =========================

results = []
frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 분석 완료.")
        break

    frame_idx += 1

    # 너무 느리면 3프레임마다 분석
    # 모든 프레임을 분석하고 싶으면 아래 두 줄을 지워도 됨
    if frame_idx % 3 != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.face_blendshapes:
        blendshapes = detection_result.face_blendshapes[0]

        data = {
            "frame": frame_idx
        }

        # MediaPipe가 뽑은 모든 표정 수치 저장
        for category in blendshapes:
            data[category.category_name] = category.score

        # 간단한 감정 점수 계산
        mouth_smile_left = data.get("mouthSmileLeft", 0)
        mouth_smile_right = data.get("mouthSmileRight", 0)
        jaw_open = data.get("jawOpen", 0)
        brow_inner_up = data.get("browInnerUp", 0)
        brow_down_left = data.get("browDownLeft", 0)
        brow_down_right = data.get("browDownRight", 0)
        mouth_frown_left = data.get("mouthFrownLeft", 0)
        mouth_frown_right = data.get("mouthFrownRight", 0)

        data["happiness"] = (mouth_smile_left + mouth_smile_right) / 2
        data["surprise"] = (jaw_open + brow_inner_up) / 2
        data["anger"] = (brow_down_left + brow_down_right) / 2
        data["sadness"] = (mouth_frown_left + mouth_frown_right + brow_inner_up) / 3

        results.append(data)

        print(f"{frame_idx}번 프레임 분석 완료")
    else:
        print(f"{frame_idx}번 프레임: 얼굴 인식 실패")


# =========================
# 5. CSV로 저장
# =========================

cap.release()

if len(results) == 0:
    print("저장할 데이터가 없습니다. 영상에서 얼굴이 잘 보이는지 확인하세요.")
else:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 저장 완료!")
    print("저장 위치:", output_csv)
    print("저장된 프레임 수:", len(df))