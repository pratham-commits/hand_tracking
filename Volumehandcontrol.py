import  cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

wCam,hCam= 640,480

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
#range : (-65,0)
volume.SetMasterVolumeLevel(-20.0, None)
minvol=volRange[0]
maxvol=volRange[1]
vol=20
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.HandDetector(detectionConfidence=0.7)


while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    if len(lmList)>3:
        print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx, cy =(x1+x2)//2 , (y1+y2)//2
        cv2.circle(img,(x1,y1),5,(0,255,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (0,255,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.circle(img, (cx, cy), 5, (0,255,255), cv2.FILLED)

        # hand range 50 - 300
        # volume range -65 - 0


        length=math.hypot(x2-x1,y2-y1)

        vol = np.interp(length,[50,150],[minvol,maxvol])

        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    else:
        print("No value detected")

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img, (50, int(vol)), (85, 400), (0, 255, 255), cv2.FILLED)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f"FPS:{int(fps)}",(40,40),cv2.FONT_HERSHEY_PLAIN
                ,2,(255,0,0),3)

    cv2.imshow("img",img)

    cv2.waitKey(1)