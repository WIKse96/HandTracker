import cv2
import mediapipe as mp
import time

import success

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handsLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handsLms.landmark):
                print(id, lm)
            mpDraw.draw_landmarks(img, handsLms, mpHands.HAND_CONNECTIONS)
            # print(handsLms)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image", img)
    cv2.waitKey(1)