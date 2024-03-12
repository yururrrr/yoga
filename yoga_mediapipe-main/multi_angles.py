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

mp_drawing = mp.solutions.drawing_utils          # mediapipe ç¹ªå????¹æ??
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe ç¹ªå??æ¨?å¼?
mp_pose = mp.solutions.pose                      # mediapipe å§¿å?¢å?µæ¸¬


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

num = '5'
#load input video
cap1 = cv2.VideoCapture('/Users/chenyuru/Desktop/Yoga/camera_test/'+num+'_right_front.MP4')
cap2 = cv2.VideoCapture('/Users/chenyuru/Desktop/Yoga/camera_test/'+num+'_front.MP4')
cap3 = cv2.VideoCapture('/Users/chenyuru/Desktop/Yoga/camera_test/'+num+'_left_front.MP4')
# Load the frame rate of the video using OpenCV???s CV_CAP_PROP_FPS
index = 0
timestamp = []

fps = int(cap1.get(cv2.CAP_PROP_FPS))

to_path = '/Users/chenyuru/Desktop/Yoga/output/camera_test/'+num

if not os.path.isdir(to_path):
    os.makedirs(to_path)


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  
    while cap1.isOpened():

        success, image = cap1.read()
        
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # landmarks data
        detection_result = results.pose_landmarks
      
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
       
        if detection_result != None :
            r_ankle.append( cal_angle(detection_result.landmark[28], detection_result.landmark[30], detection_result.landmark[32]) )
            l_ankle.append( cal_angle(detection_result.landmark[27], detection_result.landmark[29], detection_result.landmark[31]) )
            r_knee.append( cal_angle(detection_result.landmark[24], detection_result.landmark[26], detection_result.landmark[28]) )
            l_knee.append( cal_angle(detection_result.landmark[23], detection_result.landmark[25], detection_result.landmark[27]) )
            r_hip.append( cal_angle(detection_result.landmark[12], detection_result.landmark[24], detection_result.landmark[26]) )
            l_hip.append( cal_angle(detection_result.landmark[11], detection_result.landmark[23], detection_result.landmark[25]) )
            r_shoulder.append( cal_angle(detection_result.landmark[14], detection_result.landmark[12], detection_result.landmark[24]) )
            l_shoulder.append( cal_angle(detection_result.landmark[13], detection_result.landmark[11], detection_result.landmark[23]) )
            r_elbow.append( cal_angle(detection_result.landmark[16], detection_result.landmark[14], detection_result.landmark[12]) )
            l_elbow.append( cal_angle(detection_result.landmark[15], detection_result.landmark[13], detection_result.landmark[11]) )
            r_wrist.append( cal_angle(detection_result.landmark[20], detection_result.landmark[16], detection_result.landmark[14]) )
            l_wrist.append( cal_angle(detection_result.landmark[19], detection_result.landmark[15], detection_result.landmark[13]) )
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


        index += 1
        
        timestamp.append( index/fps )
        
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap1.release()

    # to csv
    pd.DataFrame({'time':timestamp,
                'l_ankle':l_ankle, 
                'r_ankle':r_ankle,
                'l_knee':l_knee,
                'r_knee':r_knee,
                'l_hip':l_hip,
                'r_hip':r_hip,
                'l_shoulder':l_shoulder,
                'r_shoulder':r_shoulder,
                'l_elbow':l_elbow,
                'r_elbow':r_elbow,
                'l_wrist':l_wrist,
                'r_wrist':r_wrist}).to_csv(to_path+'/rfront.csv')
    
    r_ankle.clear()
    l_ankle.clear()
    r_knee.clear()
    l_knee.clear()
    r_hip.clear()
    l_hip.clear()
    r_shoulder.clear()
    l_shoulder.clear()
    r_elbow.clear()
    l_elbow.clear()
    r_wrist.clear()
    l_wrist.clear()
    timestamp.clear()
    index = 0
    
    while cap2.isOpened():

        success, image = cap2.read()
        
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # landmarks data
        detection_result = results.pose_landmarks
    
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
       
        if detection_result != None :
            r_ankle.append( cal_angle(detection_result.landmark[28], detection_result.landmark[30], detection_result.landmark[32]) )
            l_ankle.append( cal_angle(detection_result.landmark[27], detection_result.landmark[29], detection_result.landmark[31]) )
            r_knee.append( cal_angle(detection_result.landmark[24], detection_result.landmark[26], detection_result.landmark[28]) )
            l_knee.append( cal_angle(detection_result.landmark[23], detection_result.landmark[25], detection_result.landmark[27]) )
            r_hip.append( cal_angle(detection_result.landmark[12], detection_result.landmark[24], detection_result.landmark[26]) )
            l_hip.append( cal_angle(detection_result.landmark[11], detection_result.landmark[23], detection_result.landmark[25]) )
            r_shoulder.append( cal_angle(detection_result.landmark[14], detection_result.landmark[12], detection_result.landmark[24]) )
            l_shoulder.append( cal_angle(detection_result.landmark[13], detection_result.landmark[11], detection_result.landmark[23]) )
            r_elbow.append( cal_angle(detection_result.landmark[16], detection_result.landmark[14], detection_result.landmark[12]) )
            l_elbow.append( cal_angle(detection_result.landmark[15], detection_result.landmark[13], detection_result.landmark[11]) )
            r_wrist.append( cal_angle(detection_result.landmark[20], detection_result.landmark[16], detection_result.landmark[14]) )
            l_wrist.append( cal_angle(detection_result.landmark[19], detection_result.landmark[15], detection_result.landmark[13]) )
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

        index += 1
        timestamp.append( index/fps )
        
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap2.release()
   
    # to csv
    pd.DataFrame({'time':timestamp,
                'l_ankle':l_ankle, 
                'r_ankle':r_ankle,
                'l_knee':l_knee,
                'r_knee':r_knee,
                'l_hip':l_hip,
                'r_hip':r_hip,
                'l_shoulder':l_shoulder,
                'r_shoulder':r_shoulder,
                'l_elbow':l_elbow,
                'r_elbow':r_elbow,
                'l_wrist':l_wrist,
                'r_wrist':r_wrist}).to_csv(to_path+'/front.csv')

    r_ankle.clear()
    l_ankle.clear()
    r_knee.clear()
    l_knee.clear()
    r_hip.clear()
    l_hip.clear()
    r_shoulder.clear()
    l_shoulder.clear()
    r_elbow.clear()
    l_elbow.clear()
    r_wrist.clear()
    l_wrist.clear()
    timestamp.clear()
    index = 0
    
    while cap3.isOpened():

        success, image = cap3.read()
        
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # landmarks data
        detection_result = results.pose_landmarks
    
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
       
        if detection_result != None :
            r_ankle.append( cal_angle(detection_result.landmark[28], detection_result.landmark[30], detection_result.landmark[32]) )
            l_ankle.append( cal_angle(detection_result.landmark[27], detection_result.landmark[29], detection_result.landmark[31]) )
            r_knee.append( cal_angle(detection_result.landmark[24], detection_result.landmark[26], detection_result.landmark[28]) )
            l_knee.append( cal_angle(detection_result.landmark[23], detection_result.landmark[25], detection_result.landmark[27]) )
            r_hip.append( cal_angle(detection_result.landmark[12], detection_result.landmark[24], detection_result.landmark[26]) )
            l_hip.append( cal_angle(detection_result.landmark[11], detection_result.landmark[23], detection_result.landmark[25]) )
            r_shoulder.append( cal_angle(detection_result.landmark[14], detection_result.landmark[12], detection_result.landmark[24]) )
            l_shoulder.append( cal_angle(detection_result.landmark[13], detection_result.landmark[11], detection_result.landmark[23]) )
            r_elbow.append( cal_angle(detection_result.landmark[16], detection_result.landmark[14], detection_result.landmark[12]) )
            l_elbow.append( cal_angle(detection_result.landmark[15], detection_result.landmark[13], detection_result.landmark[11]) )
            r_wrist.append( cal_angle(detection_result.landmark[20], detection_result.landmark[16], detection_result.landmark[14]) )
            l_wrist.append( cal_angle(detection_result.landmark[19], detection_result.landmark[15], detection_result.landmark[13]) )
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

        index += 1
        timestamp.append( index/fps )
        
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap3.release()

    # to csv
    pd.DataFrame({'time':timestamp,
                'l_ankle':l_ankle, 
                'r_ankle':r_ankle,
                'l_knee':l_knee,
                'r_knee':r_knee,
                'l_hip':l_hip,
                'r_hip':r_hip,
                'l_shoulder':l_shoulder,
                'r_shoulder':r_shoulder,
                'l_elbow':l_elbow,
                'r_elbow':r_elbow,
                'l_wrist':l_wrist,
                'r_wrist':r_wrist}).to_csv(to_path+'/lfront.csv')

