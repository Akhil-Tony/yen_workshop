import mediapipe as mp
import cv2 as cv
import numpy as np

new_main = []
new_labels = []

label_name = int(input('what should i learn '))
total_frames = 200 # <---- CHANGE FOR MAX FRAMES default 200

cam = cv.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
frame_count = 0
while True:
    success,frame = cam.read()
    if success:
        frame = cv.flip(frame,1)
        frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        processed = hands.process(frame_rgb)
        if processed.multi_hand_landmarks:
            for handLms in processed.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
                xs = []
                ys = []
                for idx,lm in enumerate(handLms.landmark):
                    xs.append(lm.x)
                    ys.append(lm.y)
                new_main.append(xs+ys)
                new_labels.append(label_name)
                frame_count+=1
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv.destroyAllWindows()
            break
        if frame_count == total_frames:
            cam.release()
            cv.destroyAllWindows()
            break

np.save('data/data',new_main)
np.save('data/labels',new_labels)
