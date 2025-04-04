import cv2
import time
import mediapipe as mp
import HandTrackingModule as htm
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()  # Create an instance of HandDetector

Ptime = 0  # Initialize previous time for FPS calculation

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)

    lmList = detector.findPosition(img)
    if len(lmList) > 5:
        print("Landmark 5:", lmList[5])
    else:
        print("Hand not fully detected ")

    # FPS Calculation
    Ctime = time.time()
    fps = 1 / (Ctime - Ptime) if Ptime != 0 else 0
    Ptime = Ctime

    # Display FPS
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show the output
    cv2.imshow("Hand Tracking", img)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()