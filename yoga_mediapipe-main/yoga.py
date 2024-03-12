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
    # 向量
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

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測


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
videos = ['1_front', '1_left_front', '1_side', '2_front', '2_left_front', '2_side', '3_front', '3_left_front', '3_side', '4_front', '4_left_front', '4_right_front', '5_front', '5_left_front', '5_right_front']

#load input video
for video in videos:
   
    
    cap = cv2.VideoCapture('~/Desktop/yuru/VIDEO/'+video+'.MP4')
    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    index = 0
    time = 0
    timestamp = []
    frame_count = 0
    downsampling_factor = 5

    # Get video properties (frame width, frame height, frames per second, etc.)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('fps=', fps)

# Create a VideoWriter object to save the output video with detections
    output_video_path = '~/Desktop/yuru/VIDEO/output/'+video+'_out.MP4'  # Replace with your desired output video file path
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    forward_fold = False

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                break

            # frame_count += 1

            # 根据下采样因子跳过一定数量的帧
            # if frame_count % downsampling_factor != 0:
            #     continue
        
            # h, w = image.shape[:2]
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # landmarks data
            detection_result = results.pose_landmarks
            # print(detection_result.landmark[30],detection_result.landmark[28], )
            # print('----------------------')
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # if index==1097:
            #     print(detection_result)
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


            # forward fold : hip, knee
            if (l_hip[index] <=40 and r_hip[index] <= 40) and (l_knee[index] >=166 and r_knee[index] >= 166):
                forward_fold = True

            
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            out.write(image)
            # cv2.imwrite('/Users/chenyuru/Desktop/Yoga/output/sample2/'+str(index)+'output.jpg',image)
            index += 1
            # if index % 60 == 0:
                # time += 1
            timestamp.append( index/fps )
            
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cap.release()
        out.release()

    # print(index)
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
                'r_wrist':r_wrist}).to_csv('~/Desktop/yuru/VIDEO/output/csv/'+video+'_out.csv')
