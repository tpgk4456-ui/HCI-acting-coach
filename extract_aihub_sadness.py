import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import random
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. 기본 설정
# =========================

# MediaPipe Face Landmarker 모델 파일 경로
model_path = "face_landmarker.task"

# AI-Hub 데이터가 저장된 기본 폴더 경로
# AI-Hub 데이터가 저장된 기본 폴더 경로
base_dir = Path(r"D:\Seoyeon\HCI\AIHub\korean_emotion_validation")

# blendshape를 뽑을 실제 이미지 폴더는 raw 안에 있음
image_root_dir = base_dir / "raw"

# 슬픔 이미지 폴더
emotion_dirs = {
    "sadness": image_root_dir / "EMOIMG_슬픔_VALID"
}

# 감정별 샘플링할 이미지 개수
sample_size_per_emotion = 500

# 랜덤 샘플링 고정값
random_seed = 316

# 전체 blendshape 결과 저장 파일
output_csv = "aihub_sadness_blendshapes.csv"

# 슬픔 분포 요약 저장 파일
sadness_distribution_csv = "sadness_distribution.csv"


# =========================
# 2. MediaPipe Face Landmarker 설정
# =========================

# 모델 파일이 실제로 있는지 확인
if not Path(model_path).exists():
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

# MediaPipe 모델 옵션 설정
base_options = python.BaseOptions(model_asset_path=model_path)

# 얼굴 blendshape 출력을 켜는 설정
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

# Face Landmarker 생성
detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# 3. 유틸 함수
# =========================

def get_image_paths(image_dir):
    # 이미지 폴더를 Path 객체로 변환
    image_dir = Path(image_dir)

    # 폴더가 실제로 있는지 확인
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    # jpg, jpeg, png 이미지를 하위 폴더까지 모두 찾기
    image_paths = []
    image_paths.extend(image_dir.rglob("*.jpg"))
    image_paths.extend(image_dir.rglob("*.jpeg"))
    image_paths.extend(image_dir.rglob("*.png"))
    image_paths.extend(image_dir.rglob("*.JPG"))
    image_paths.extend(image_dir.rglob("*.JPEG"))
    image_paths.extend(image_dir.rglob("*.PNG"))

    # 중복 제거 후 정렬
    image_paths = sorted(list(set(image_paths)))

    return image_paths


def read_image_unicode(image_path):
    # Windows에서 한글 경로를 안정적으로 읽기 위해 np.fromfile 사용
    image_array = np.fromfile(str(image_path), dtype=np.uint8)

    # 이미지 파일을 OpenCV 이미지로 디코딩
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return frame


def extract_blendshape_dict(blendshapes):
    # MediaPipe blendshape 결과를 dictionary로 변환
    data = {}

    # category_name은 browDownLeft, jawOpen 같은 blendshape 이름
    for category in blendshapes:
        data[category.category_name] = float(category.score)

    return data


def detect_image(image_path):
    # 한글 경로에서도 읽히도록 이미지 로드
    frame = read_image_unicode(image_path)

    # 이미지 읽기에 실패하면 실패 원인을 함께 반환
    if frame is None:
        return None, "image_read_failed"

    # OpenCV는 BGR이므로 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 이미지 객체 생성
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # 얼굴 감지 및 blendshape 추출
    detection_result = detector.detect(mp_image)

    # 얼굴 감지 실패
    if not detection_result.face_landmarks:
        return None, "face_not_detected"

    # blendshape 추출 실패
    if not detection_result.face_blendshapes:
        return None, "blendshape_not_detected"

    # 첫 번째 얼굴의 blendshape만 사용
    blendshapes = detection_result.face_blendshapes[0]

    return extract_blendshape_dict(blendshapes), "success"


def sample_images(image_paths, sample_size):
    # 이미지 개수가 sample_size보다 작으면 전체 사용
    if len(image_paths) <= sample_size:
        return image_paths

    # 랜덤하게 sample_size개 선택
    return random.sample(image_paths, sample_size)


def process_emotion_images(image_dir, emotion_label, sample_size):
    # 폴더 안의 이미지 경로 전체 찾기
    image_paths = get_image_paths(image_dir)

    # 이미지 개수 출력
    print(f"[{emotion_label}] 전체 이미지 수: {len(image_paths)}")

    # 이미지가 하나도 없으면 바로 종료
    if len(image_paths) == 0:
        raise RuntimeError(f"[{emotion_label}] 이미지가 없습니다. 폴더 경로를 확인하세요: {image_dir}")

    # 이미지 경로 예시 출력
    print(f"[{emotion_label}] 이미지 예시 3개")
    for example_path in image_paths[:3]:
        print(" -", example_path)

    # 랜덤 샘플링
    sampled_paths = sample_images(image_paths, sample_size)

    # 샘플링 개수 출력
    print(f"[{emotion_label}] 샘플링 이미지 수: {len(sampled_paths)}")

    # 결과를 저장할 리스트
    results = []

    # 실패 개수 카운트
    failed_count = 0

    # 실패 원인별 카운트
    failed_reasons = {
        "image_read_failed": 0,
        "face_not_detected": 0,
        "blendshape_not_detected": 0
    }

    # 샘플 이미지들을 하나씩 처리
    for idx, image_path in enumerate(sampled_paths, start=1):
        # 이미지에서 blendshape 추출
        blendshape_data, status = detect_image(image_path)

        # 실패 시 원인 기록
        if blendshape_data is None:
            failed_count += 1

            # 실패 원인 카운트
            if status in failed_reasons:
                failed_reasons[status] += 1
            else:
                failed_reasons[status] = 1

            # 처음 10개 실패만 자세히 출력
            if failed_count <= 10:
                print(f"[{emotion_label} 실패] {status}: {image_path}")

            continue

        # 기본 메타데이터 추가
        row = {
            "emotion": emotion_label,
            "image_path": str(image_path),
            "image_name": image_path.name
        }

        # blendshape 값 추가
        row.update(blendshape_data)

        # 결과 리스트에 저장
        results.append(row)

        # 진행 상황 출력
        if idx % 50 == 0:
            print(f"[{emotion_label}] 진행: {idx}/{len(sampled_paths)}")

    # 완료 로그 출력
    print(f"[{emotion_label}] 성공: {len(results)}개, 실패: {failed_count}개")
    print(f"[{emotion_label}] 실패 원인별 개수: {failed_reasons}")

    return results


def save_distribution(df, emotion_label, output_path):
    # 특정 감정 데이터만 선택
    emotion_df = df[df["emotion"] == emotion_label].copy()

    # 해당 감정 데이터가 없으면 저장하지 않음
    if len(emotion_df) == 0:
        print(f"{emotion_label} 데이터가 없어서 분포 파일을 저장하지 않습니다.")
        return

    # 메타데이터 컬럼 제외
    meta_cols = ["emotion", "image_path", "image_name"]

    # blendshape 컬럼만 선택
    blendshape_cols = [col for col in emotion_df.columns if col not in meta_cols]

    # 감정별 분포 요약 생성
    distribution = pd.DataFrame({
        "blendshape": blendshape_cols,
        "mean": emotion_df[blendshape_cols].mean().values,
        "std": emotion_df[blendshape_cols].std().values,
        "median": emotion_df[blendshape_cols].median().values,
        "q25": emotion_df[blendshape_cols].quantile(0.25).values,
        "q75": emotion_df[blendshape_cols].quantile(0.75).values
    })

    # CSV 저장
    distribution.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 저장 로그 출력
    print(f"{emotion_label} 분포 저장 완료: {output_path}")


# =========================
# 4. 실행
# =========================

# 랜덤 seed 고정
random.seed(random_seed)

# 폴더 확인 로그
print("AI-Hub 기본 폴더:", base_dir)
print("이미지 상위 폴더:", image_root_dir)

for emotion_label, emotion_dir in emotion_dirs.items():
    print(f"{emotion_label} 폴더 존재:", emotion_dir.exists())
    print(f"{emotion_label} 폴더 경로:", emotion_dir)

# 전체 결과를 저장할 리스트
all_results = []

# 감정별 이미지 처리
for emotion_label, emotion_dir in emotion_dirs.items():
    emotion_results = process_emotion_images(
        emotion_dir,
        emotion_label,
        sample_size_per_emotion
    )

    all_results.extend(emotion_results)

# 결과가 하나도 없으면 종료
if len(all_results) == 0:
    raise RuntimeError(
        "추출된 blendshape 결과가 없습니다. "
        "이미지는 읽혔는지, MediaPipe가 얼굴을 감지했는지 실패 원인 로그를 확인하세요."
    )

# DataFrame으로 변환
df = pd.DataFrame(all_results)

# 전체 blendshape 결과 저장
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

# 저장 로그 출력
print(f"\n전체 blendshape CSV 저장 완료: {output_csv}")
print(f"총 저장 개수: {len(df)}")

# 슬픔 분포 저장
save_distribution(df, "sadness", sadness_distribution_csv)

# 감정별 저장 개수 출력
print("\n감정별 저장 개수")
print(df["emotion"].value_counts())