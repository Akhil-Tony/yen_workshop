import time
import cv2 as cv
import mediapipe as mp
from tensorflow import keras
import numpy as np

def preprocess(x,y):
    joined =  x+y
    joined_arr = np.array(joined)
    return joined_arr

def findLabel(prediction):
    dominant_pred = np.argmax(prediction)
    label = class_label[dominant_pred]
    prob = prediction[0][dominant_pred]
    prob = np.round(prob,2)
    return label,prob

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

class_label = ['Hai','Well Done','Victory','Fuck Off']
gesture_model = keras.models.load_model('best_weight.h5')

ctime = 0
ptime = 0
cam0 = cv.VideoCapture(1)

while True:
    success,frame = cam0.read()

    if success:
        frame = cv.flip(frame,1)
        h,w,c = frame.shape
        frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                xs = []
                ys = []
                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
                for idx,lm in enumerate(handLms.landmark):
                    xs.append(lm.x)
                    ys.append(lm.y)
                processed_instance = preprocess(xs,ys)
                prediction = gesture_model.predict(processed_instance.reshape(1,42))
                label,prob = findLabel(prediction)
                cv.putText(frame,'Detection : '+label,(7,120),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2) # + '['+str(prob)+'%]'
                
        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime
        cv.putText(frame,'fps : '+str(fps),(7,30),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cam0.release()
cv.destroyAllWindows()
