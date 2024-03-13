import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from dataset import VideoDataset

def cal_angle(point_a , point_b, point_c):
    x1, y1, z1 = (point_a.x - point_b.x), (point_a.y - point_b.y), (point_a.z - point_b.z)
    x2, y2, z2 = (point_c.x - point_b.x), (point_c.y - point_b.y), (point_c.z - point_b.z)
    
    tmp = (math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2))
    if tmp == 0:
        B = -1
    else:
        cos_b = (x1*x2 + y1*y2 + z1*z2) / tmp
        B = math.degrees(math.acos(cos_b))

    return B

model_path = 'pose_landmark.task'

mp_drawing = mp.solutions.drawing_utils        
mp_drawing_styles = mp.solutions.drawing_styles  
mp_pose = mp.solutions.pose                     

points = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist']


home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
data_path = home_path+'yoga-pose-classification/dataset_mat'

def detect_keypoints(img_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)
    
    # keypoints = []
    # if results.pose_landmarks:
    #     for landmark in results.pose_landmarks.landmark:
    #         keypoints.append((landmark.x, landmark.y, landmark.z))

    return results.pose_landmarks


labels = sorted(os.listdir(data_path))
for label in labels:
    label_path = os.path.join(data_path, label)
    videos = os.listdir(label_path)
    for video in videos:
        video_path = os.path.join(label_path, video)
        frames = os.listdir(video_path)
        joints = {r_ankle:[], l_ankle:[], r_knee:[], l_knee:[], r_hip:[], l_hip:[],
                  r_shoulder:[], l_shoulder:[], r_elbow:[], l_elbow:[], r_wrist:[], l_wrist:[]}
        r_ankle = []
        l_ankle = []
        r_knee = []
        l_knee = []
        r_hip = []
        l_hip = []
        r_shoulder = []
        l_shoulder = []
        r_elbow = []
        l_elbow = []
        r_wrist = []
        l_wrist = []
        
        for frame in frames:
            frame_path = os.path.join(video_path, frame)
            keypoint = detect_keypoints(frame_path)
            
            if keypoint :
                # print(keypoint[28])
                r_ankle.append( cal_angle(keypoint.landmark[28], keypoint.landmark[30], keypoint.landmark[32]) )
                l_ankle.append( cal_angle(keypoint.landmark[27], keypoint.landmark[29], keypoint.landmark[31]) )
                r_knee.append( cal_angle(keypoint.landmark[24], keypoint.landmark[26], keypoint.landmark[28]) )
                l_knee.append( cal_angle(keypoint.landmark[23], keypoint.landmark[25], keypoint.landmark[27]) )
                r_hip.append( cal_angle(keypoint.landmark[12], keypoint.landmark[24], keypoint.landmark[26]) )
                l_hip.append( cal_angle(keypoint.landmark[11], keypoint.landmark[23], keypoint.landmark[25]) )
                r_shoulder.append( cal_angle(keypoint.landmark[14], keypoint.landmark[12], keypoint.landmark[24]) )
                l_shoulder.append( cal_angle(keypoint.landmark[13], keypoint.landmark[11], keypoint.landmark[23]) )
                r_elbow.append( cal_angle(keypoint.landmark[16], keypoint.landmark[14], keypoint.landmark[12]) )
                l_elbow.append( cal_angle(keypoint.landmark[15], keypoint.landmark[13], keypoint.landmark[11]) )
                r_wrist.append( cal_angle(keypoint.landmark[20], keypoint.landmark[16], keypoint.landmark[14]) )
                l_wrist.append( cal_angle(keypoint.landmark[19], keypoint.landmark[15], keypoint.landmark[13]) )
            else:
                r_ankle.append(-1)
                l_ankle.append(-1)
                r_knee.append(-1)
                l_knee.append(-1)
                r_hip.append(-1)
                l_hip.append(-1)
                r_shoulder.append(-1)
                l_shoulder.append(-1)
                r_elbow.append(-1)
                l_elbow.append(-1)
                r_wrist.append(-1)
                l_wrist.append(-1)

        
        print(f'finished {video} in {label}')