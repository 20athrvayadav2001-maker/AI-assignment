import cv2
import random

video = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width, _ = frame.shape

    x = random.randint(50, width - 300)
    y = random.randint(50, height - 100)

    cv2.putText(
        frame,
        "Automotive Intelligence",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "python assignment, version 0",
        (20, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Python Assignment - Part 0", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()