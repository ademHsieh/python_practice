import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (0, 0, 255), thickness = 5)  #更改點的樣式
handConStyle = mpDraw.DrawingSpec(color = (0, 255, 0), thickness = 10)  #更改線的樣式
pTime = 0
cTime = 0


while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)   #將手上的線連接起來
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)    #int 輸出座標為整數
                    yPos = int(lm.y * imgHeight)
                    #cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2) #在點旁加上編號
                    if i == 8:    #第八個點為食指
                        cv2.circle(img, (xPos, yPos), 15, (255, 0, 0), cv2.FILLED)
                    
                    print(i, xPos, yPos)

        #顯示FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)

        cv2.imshow('img', img)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break


