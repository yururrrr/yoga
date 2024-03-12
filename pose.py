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

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


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

home_path = 'C:/Users/VIPLAB/Desktop/yuru'
data_path = home_path+'yoga-pose-classification/dataset_mat'

dataset = VideoDataset(root_dir=data_path, transform=None)
num_classes = len(dataset.labels)
