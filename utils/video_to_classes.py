import cv2
import numpy as np
import os
import pandas as pd

videos = [ '1_front', '1_side', '1_left_front', '2_front', '2_left_front', '2_side', '3_front', '3_left_front', '3_side', '4_front', '4_left_front', '4_right_front', '5_front', '5_left_front', '5_right_front']
# videos = [ '2_front', '2_left_front','3_front', '3_left_front', '3_side','5_front', '5_left_front', '5_right_front']

# '1_front', '1_left_front', '1_side', 
# videos = ['test5']
labels = ['1.mountain pose', '2.raised hands pose', '3.standing forward bend', '4.half standing forward bend', '5.plank', '6.four-limbed staff pose', '7.upward facing dog', '8.downward facing dog']

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

HOME_PATH = 'E:/yoga/'
video_path = HOME_PATH+'VIDEO/crop/'#'RobustVideoMatting-master/RobustVideoMatting-master/output/'
data_path = HOME_PATH+'yoga-pose-classification/'

OUT_PATH = check_dir(data_path + 'dataset_crop/')

for la in labels: check_dir(OUT_PATH+la)

df = pd.read_excel(data_path+'movement_frames_info.xlsx', sheet_name='train')
    

# video = videos[1]
v_num = 0
for video in videos:    
        
    cap = cv2.VideoCapture(video_path + video + '.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open video {video}")
        continue
    
    rows = df.loc[df['VIDEO_name'] == video]
    rows.reset_index(drop=True)
    index = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        output_frame_path = ''
        if count >= rows.iloc[index, 2] and count <= rows.iloc[index, 3]:
            pose = rows.iloc[index, 1]
            # print(pose)
            
            if pose ==1:    output_frame_path = f'1.mountain pose/{v_num}/'
            elif pose ==2:  output_frame_path = f'2.raised hands pose/{v_num}/'
            elif pose ==3:  output_frame_path = f'3.standing forward bend/{v_num}/'
            elif pose==4:   output_frame_path = f'4.half standing forward bend/{v_num}/'
            elif pose==5:   output_frame_path = f'5.plank/{v_num}/'
            elif pose==6:   output_frame_path = f'6.four-limbed staff pose/{v_num}/'
            elif pose==7:   output_frame_path = f'7.upward facing dog/{v_num}/'
            elif pose==8:   output_frame_path = f'8.downward facing dog/{v_num}/'
        
            if count==rows.iloc[index, 3] : 
                index+=1
                v_num+=1
                print(f'fisnied video {video} pose={pose}')
            if index==12: 
                print(f'----------finished video {video}----------')
                break
            out = os.path.join(OUT_PATH, output_frame_path)
            check_dir(out)
            
            cv2.imwrite(f'{out}/{video}_{count}.png', frame)
        # print(output_frame_path)
        # breakcvxcvv
        count += 1 
        # cv2.imwrite(output_frame_path, frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break
        
    cap.release()
    
    