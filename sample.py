import cv2 as cv
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

cam = cv.VideoCapture(1)

while True:
    success,frame = cam.read()

    if success:
        frame = cv.flip(frame,1)
        h,w,c = frame.shape
        frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
                
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv.destroyAllWindows()
