import cv2
import time
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        # Initialize Mediapipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionConfidence,
                                        min_tracking_confidence=self.trackConfidence)

        # Initialize Mediapipe drawing utilities
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            MyHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(MyHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw :  # Only draw circle on landmark 7
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()  # Create an instance of HandDetector

    Ptime = 0  # Initialize previous time for FPS calculation

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.findHands(img)

        lmList=detector.findPosition(img)
        if len(lmList) > 5:
            print("Landmark 5:", lmList[5])
        else:
            print("Hand not fully detected or landmark 5 missing")

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


if __name__ == "__main__":
    main()
