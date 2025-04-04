

import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)
#to detect the hands
mpHands=mp.solutions.hands
hands=mpHands.Hands()

#to draw the lines
mpDraw=mp.solutions.drawing_utils
Ctime=0
Ptime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    #if hands are present in the screen
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                #for printing the id of the 20 areas on the palm
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #showing the where the first id belongs on the palm
                if(id==15):
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    Ctime=time.time()
    fps=1/(Ctime-Ptime)
    Ptime=Ctime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)
