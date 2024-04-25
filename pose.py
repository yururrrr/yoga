import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from scipy.signal import savgol_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='video folder path')
    parser.add_argument('--save_path', type=str, help='save folder path')

    return parser.parse_args()


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
       
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def detect_keypoints(img):
    
    # img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    return results.pose_landmarks

def plot_points(data, point, path, rows):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
    y_smooth = savgol_filter(data, 30, 2)
    
    plt.plot(data, alpha=0.2, color='b')
    plt.plot(y_smooth, color='b')
    plt.xlabel('time')
    plt.ylabel('Position')
    plt.title(point, fontsize=30)

    # print(' not in loop')
    for i in range(len(rows)):
        # print('in loop')
        pose = rows.loc[i, 'Pose']
        start = rows.loc[i, 'start']
        end = rows.loc[i, 'end']
        # print('in row')
        plt.axvline(x=start, ls='--') #start

        plt.axvline(x=end, ls='--') #start
        plt.text(start, -1, f'Pose{pose}',fontsize=20)

    plt.savefig(f'{path}/{point}.png')
    plt.clf()
    plt.close()
    print('finished plot')
    return 


def main():

    args = parse_args()
    home_path = 'C:/Users/VIP/Desktop/yuru/yoga/'
    data_path = os.path.join(home_path, args.data_path)
    save_path = os.path.join(home_path, args.save_path)

    if args.data_path is None or not os.path.exists(data_path):
        print("Please provide a valid data path.")
        print(data_path)
        return
    os.makedirs(save_path, exist_ok=True)

    video_files = [f for f in os.listdir(data_path) if f.endswith('.MP4') ]#and f!='1_front.MP4'
    # print(video_files)
 
    df = pd.read_excel(home_path+'yoga-pose-classification/movement_frames_info.xlsx', sheet_name='train')

    points = ['R_ankle', 'L_ankle', 'R_knee', 'L_knee', 'R_hip', 'L_hip', 'R_shoulder', 'L_shoulder', 'R_elbow', 'L_elbow', 'R_wrist', 'L_wrist']

    for video in video_files:
        print(video)
        cap = cv2.VideoCapture(os.path.join(data_path, video))
        if not cap.isOpened():
            print(f"Error: Could not opne video {video}")
            continue

        video_name = os.path.splitext(video)[0]
        rows = df.loc[df['VIDEO_name'] == video_name]
        rows.reset_index(drop=True)

        points_info = {'R_ankle':[], 'L_ankle': [], 'R_knee': [], 'L_knee': [], 'R_hip': [], 'L_hip': [],
                        'R_shoulder': [], 'L_shoulder': [], 'R_elbow': [], 'L_elbow': [], 'R_wrist': [], 'L_wrist': []}
        # print('here')
        count=0
        while cap.isOpened():
            
            ret, frame = cap.read()
            # fps = int(cap.get(cv2.CAP_PROP_FPS))
            if not ret:
                print("Ignoring empty camera frame.")
                break
            keypoint = detect_keypoints(frame)

            if keypoint :
                # print(keypoint[28])
                count+=1
                # points_info['R_ankle'].append( cal_angle(keypoint.landmark[28], keypoint.landmark[30], keypoint.landmark[32]) )
                # points_info['L_ankle'].append( cal_angle(keypoint.landmark[27], keypoint.landmark[29], keypoint.landmark[31]) )
                # points_info['R_knee'].append( cal_angle(keypoint.landmark[24], keypoint.landmark[26], keypoint.landmark[28]) )
                # points_info['L_knee'].append( cal_angle(keypoint.landmark[23], keypoint.landmark[25], keypoint.landmark[27]) )
                # points_info['R_hip'].append( cal_angle(keypoint.landmark[12], keypoint.landmark[24], keypoint.landmark[26]) )
                # points_info['L_hip'].append( cal_angle(keypoint.landmark[11], keypoint.landmark[23], keypoint.landmark[25]) )
                # points_info['R_shoulder'].append( cal_angle(keypoint.landmark[14], keypoint.landmark[12], keypoint.landmark[24]) )
                # points_info['L_shoulder'].append( cal_angle(keypoint.landmark[13], keypoint.landmark[11], keypoint.landmark[23]) )
                # points_info['R_elbow'].append( cal_angle(keypoint.landmark[16], keypoint.landmark[14], keypoint.landmark[12]) )
                # points_info['L_elbow'].append( cal_angle(keypoint.landmark[15], keypoint.landmark[13], keypoint.landmark[11]) )
                # points_info['R_wrist'].append( cal_angle(keypoint.landmark[20], keypoint.landmark[16], keypoint.landmark[14]) )
                # points_info['L_wrist'].append( cal_angle(keypoint.landmark[19], keypoint.landmark[15], keypoint.landmark[13]) )
                points_info['R_ankle'].append( keypoint.landmark[28].x )
                points_info['L_ankle'].append( keypoint.landmark[27].x )
                points_info['R_knee'].append( keypoint.landmark[26].x )
                points_info['L_knee'].append( keypoint.landmark[25].x )
                points_info['R_hip'].append( keypoint.landmark[24].x )
                points_info['L_hip'].append( keypoint.landmark[23].x )
                points_info['R_shoulder'].append( keypoint.landmark[12].x )
                points_info['L_shoulder'].append( keypoint.landmark[11].x )
                points_info['R_elbow'].append( keypoint.landmark[14].x )
                points_info['L_elbow'].append( keypoint.landmark[13].x )
                points_info['R_wrist'].append( keypoint.landmark[16].x )
                points_info['L_wrist'].append( keypoint.landmark[15].x)
            else:
                points_info['R_ankle'].append(-1)
                points_info['L_ankle'].append(-1)
                points_info['R_knee'].append(-1)
                points_info['L_knee'].append(-1)
                points_info['R_hip'].append(-1)
                points_info['L_hip'].append(-1)
                points_info['R_shoulder'].append(-1)
                points_info['L_shoulder'].append(-1)
                points_info['R_elbow'].append(-1)
                points_info['L_elbow'].append(-1)
                points_info['R_wrist'].append(-1)
                points_info['L_wrist'].append(-1)

        cap.release()
        print(f'frames={count}')
        for point in points:
            # print(points_info[point][:10])
            plot_points(points_info[point], point, os.path.join(save_path, video_name), rows)

if __name__ =='__main__':
    main()





# labels = sorted(os.listdir(data_path))
# for label in labels:
    # label_path = os.path.join(data_path, label)
    # videos = os.listdir(label_path)
    # for video in videos:
    #     video_path = os.path.join(label_path, video)
    #     frames = os.listdir(video_path)
    #     joints = {r_ankle:[], l_ankle:[], r_knee:[], l_knee:[], r_hip:[], l_hip:[],
    #               r_shoulder:[], l_shoulder:[], r_elbow:[], l_elbow:[], r_wrist:[], l_wrist:[]}
    #     r_ankle = []
    #     l_ankle = []
    #     r_knee = []
    #     l_knee = []
    #     r_hip = []
    #     l_hip = []
    #     r_shoulder = []
    #     l_shoulder = []
    #     r_elbow = []
    #     l_elbow = []
    #     r_wrist = []
    #     l_wrist = []
        
    #     for frame in frames:
    #         frame_path = os.path.join(video_path, frame)
    #         keypoint = detect_keypoints(frame_path)
            
    #         if keypoint :
    #             # print(keypoint[28])
    #             r_ankle.append( cal_angle(keypoint.landmark[28], keypoint.landmark[30], keypoint.landmark[32]) )
    #             l_ankle.append( cal_angle(keypoint.landmark[27], keypoint.landmark[29], keypoint.landmark[31]) )
    #             r_knee.append( cal_angle(keypoint.landmark[24], keypoint.landmark[26], keypoint.landmark[28]) )
    #             l_knee.append( cal_angle(keypoint.landmark[23], keypoint.landmark[25], keypoint.landmark[27]) )
    #             r_hip.append( cal_angle(keypoint.landmark[12], keypoint.landmark[24], keypoint.landmark[26]) )
    #             l_hip.append( cal_angle(keypoint.landmark[11], keypoint.landmark[23], keypoint.landmark[25]) )
    #             r_shoulder.append( cal_angle(keypoint.landmark[14], keypoint.landmark[12], keypoint.landmark[24]) )
    #             l_shoulder.append( cal_angle(keypoint.landmark[13], keypoint.landmark[11], keypoint.landmark[23]) )
    #             r_elbow.append( cal_angle(keypoint.landmark[16], keypoint.landmark[14], keypoint.landmark[12]) )
    #             l_elbow.append( cal_angle(keypoint.landmark[15], keypoint.landmark[13], keypoint.landmark[11]) )
    #             r_wrist.append( cal_angle(keypoint.landmark[20], keypoint.landmark[16], keypoint.landmark[14]) )
    #             l_wrist.append( cal_angle(keypoint.landmark[19], keypoint.landmark[15], keypoint.landmark[13]) )
    #         else:
    #             r_ankle.append(-1)
    #             l_ankle.append(-1)
    #             r_knee.append(-1)
    #             l_knee.append(-1)
    #             r_hip.append(-1)
    #             l_hip.append(-1)
    #             r_shoulder.append(-1)
    #             l_shoulder.append(-1)
    #             r_elbow.append(-1)
    #             l_elbow.append(-1)
    #             r_wrist.append(-1)
    #             l_wrist.append(-1)

        
    #     print(f'finished {video} in {label}')
