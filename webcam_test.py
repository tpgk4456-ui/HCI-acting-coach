import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("0번 카메라를 열 수 없습니다. 1번 카메라를 시도해보세요.")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 열기 성공! q를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()