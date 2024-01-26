import mediapipe as mp
import cv2 as cv
import numpy as np
import os
from scipy.spatial.distance import euclidean

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

data_points = []
data_labels = []

total_frames = 200 # <---- CHANGE FOR MAX FRAMES default 200
video_root = 'train_videos/'

for video_idx,video in enumerate(os.listdir(video_root)):
    cam = cv.VideoCapture(video_root+video)
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
                    
                    new_features = feature_engineering(handLms.landmark) #new
                    
                    data_points.append(new_features)
                    data_labels.append(video_idx)
                    frame_count+=1
            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count == total_frames:
                break
        else:
            break
    print(video.split('.')[0],video_idx,frame_count)
    cam.release()
    cv.destroyAllWindows()
print(data_labels)
np.save('data/data',data_points)
np.save('data/labels',data_labels)