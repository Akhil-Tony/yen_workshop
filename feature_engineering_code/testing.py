import time
import cv2 as cv
import mediapipe as mp
from tensorflow import keras
import numpy as np
from scipy.spatial.distance import euclidean

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
# new code
def feature_engineering(landmarks):
    finger_tips = [4,8,12,16,20]
    distance_list = [] # new features
    for finger in finger_tips:
        pt1 = (landmarks[finger].x,landmarks[finger].y)
        pt2 = (landmarks[0].x,landmarks[0].y) # wrist point
        distance = euclidean(pt1,pt2)
        distance_list.append(distance)
    
    return distance_list


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

class_label = ['hai','love','peace']
gesture_model = keras.models.load_model('gesture_model_new.h5')

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
                # new code
                processed_instance = feature_engineering(handLms.landmark)
                processed_instance = np.array(processed_instance)
                
                prediction = gesture_model.predict(processed_instance.reshape(1,5))
                label,prob = findLabel(prediction)
                cv.putText(frame,'Detection : '+label+ '['+str(prob)+'%]',(7,120),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2) # 
                
        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime
        cv.putText(frame,'fps : '+str(fps),(7,30),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cam0.release()
cv.destroyAllWindows()
