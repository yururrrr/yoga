import cv2
import mediapipe as mp
# from utiles import draw_landmarks_on_image
import numpy as np
import os
import pandas as pd

videos = [ '1_front', '1_left_front', '1_side', '2_front', '2_left_front', '2_side', '3_front', '3_left_front', '3_side', '4_front', '4_left_front', '4_right_front', '5_front', '5_left_front', '5_right_front']
# '1_front', '1_left_front', '1_side', 
videos = ['test5']
# mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
# mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
pose = mp_pose.Pose()

HOME_PATH = 'C:/Users/VIP/Desktop/yuru/'
video_path = HOME_PATH+'RobustVideoMatting-master/RobustVideoMatting-master/output/'


# video = videos[1]
for video in videos:
    cap = cv2.VideoCapture(video_path + video + '.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open video {video}")
        continue
    OUT_PATH = HOME_PATH+'output/'+video+'/'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # Get video properties (frame width, frama height, frames per second, etc.)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'video={video} : width={frame_width}, height={frame_height}, fps={fps}')

    # output_video_path = OUT_PATH + video + '_out.MP4'  # Replace with your desired output video file path
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
   
    # List to store landmarks for each frame
    all_landmarks = []
    i=0
    with mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
        while cap.isOpened():
            success, frame = cap.read()
        
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            frame.flags.writeable = False
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            detection_result = results.pose_landmarks
            # keypoints = results.pose_landmarks
            # Extract joints data
            landmarks = []
            
            if detection_result:
                landmarks = np.array( [ [landmark.x, landmark.y, landmark.z] for landmark in detection_result.landmark])
                # landmarks = landmarks.ravel()
                # if i==0:
                #     # print(landmarks.flatten())
                #     i+=1
                all_landmarks.append(landmarks.flatten())
            
            # Draw the pose annotation on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # out.write(image)
            # Save the original frame to a file (without landmarks)
            output_frame_path = f'{OUT_PATH}{video}_{len(all_landmarks)}.png'
            cv2.imwrite(output_frame_path, frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break
        
    cap.release()
    df = pd.DataFrame(all_landmarks, columns=['nose_x', 'nose_y', 'nose_z', 'L_eye_inner_x', 'L_eye_inner_y', 'L_eye_inner_z', 'L_eye_x', 'L_eye_y', 'L_eye_z', 'L_ete_outer_x', 'L_ete_outer_y', 'L_ete_outer_z',
                                              'R_eye_inner_x', 'R_eye_inner_y', 'R_eye_inner_z', 'R_eye_x', 'R_eye_y', 'R_eye_z', 'R_ete_outer_x', 'R_ete_outer_y', 'R_ete_outer_z',
                                              'L_ear_x', 'L_ear_y', 'L_ear_z', 'R_ear_x', 'R_ear_y', 'R_ear_z', 'mouth_L_x', 'mouth_L_y', 'mouth_L_z', 'mouth_R_x', 'mouth_R_y', 'mouth_R_z',
                                              'L_shoudler_x', 'L_shoudler_y', 'L_shoudler_z', 'R_shoudler_x', 'R_shoudler_y', 'R_shoudler_z', 'L_elbow_x', 'L_elbow_y', 'L_elbow_z', 'R_elbow_x', 'R_elbow_y', 'R_elbow_z',
                                              'L_wrist_x', 'L_wrist_y', 'L_wrist_z', 'R_wrist_x', 'R_wrist_y', 'R_wrist_z', 'L_pinky_x', 'L_pinky_y', 'L_pinky_z', 'R_pinky_x', 'R_pinky_y', 'R_pinky_z',
                                              'L_index_x', 'L_index_y', 'L_index_z', 'R_index_x', 'R_index_y', 'R_index_z', 'L_thumb_x', 'L_thumb_y', 'L_thumb_z', 'R_thumb_x', 'R_thumb_y', 'R_thumb_z',
                                              'L_hip_x', 'L_hip_y', 'L_hip_z', 'R_hip_x', 'R_hip_y', 'R_hip_z', 'L_knee_x', 'L_knee_y', 'L_knee_z', 'R_knee_x', 'R_knee_y', 'R_knee_z',
                                              'L_ankle_x', 'L_ankle_y', 'L_ankle_z', 'R_ankle_x', 'R_ankle_y', 'R_ankle_z', 'L_heel_x', 'L_heel_y', 'L_heel_z', 'R_heel_x', 'R_heel_y', 'R_heel_z',
                                              'L_foot_index_x', 'L_foot_index_y', 'L_foot_index_z', 'R_foot_index_x', 'R_foot_index_y', 'R_foot_index_z',])
    # print(df[:2])
    df.to_csv(OUT_PATH+'landmarks.csv', index=False)

    print('----------------')

cv2.destroyAllWindows()