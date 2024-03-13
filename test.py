import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models import CNN_LSTM_frame
# import mediapipe as mp

# num_classes = 8
# cnn_hidden_size = 512
# lstm_hidden_size = 256
CNN_model = CNN_LSTM_frame(num_classes=8, cnn_hidden_size=512, lstm_hidden_size=256)
CNN_model.load_state_dict(torch.load('model/CNN_LSTM_mat.pt'))
CNN_model.eval()

video_name = ['test3_mat']
home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
video_path = home_path+'VIDEO/test3_mat.mp4'

cap = cv2.VideoCapture(video_path)

frame_count = 0
start_time = None
pose_sequence = []
min_time_threshold = 10
min_frame_threshlod = 30

def all_same(sequence):
    return all(x == sequence[0] for x in sequence)

from dataset import VideoReader, VideoWriter
reader = VideoReader(video_path, transform=ToTensor())
writer = VideoWriter(f'{home_path}output/predicted_{video_name[0]}.mp4', frame_rate=30)

with torch.no_grad():
    for src in DataLoader(reader):
        output = CNN_model(src)
        _, predict = torch.max(output, 1)
        predicted_pose = predict.item()
        
        frame_count += 1
        pose_sequence.append(predicted_pose)
        
        if len(pose_sequence) >= min_frame_threshlod:
            if all_same(pose_sequence):
                # src to numpy
                src_numpy = src.squeeze().permute(1, 2, 0).numpy()
                src_mat = cv2.UMat(src_numpy)
                
                text = str(predicted_pose)
                cv2.putText(src_mat, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                writer.write(src_mat)
                pose_sequence = []
#         
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         print('Ignore empty camera frame.')
#         break
#     tensor_frame = transforms.ToTensor()(frame).unsqueeze(0)
#     with torch.no_grad():
#         output = model(tensor_frame)
#         _, predict = torch.max(output, 1)
#         predicted_pose = predict.item()
        
#     frame_count += 1
#     pose_sequence.append(predicted_pose)
    
#     if start_time is None:
#         start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
#     else:
#         current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
#         time_difference = current_time - start_time 
#         if time_difference >= min_time_threshold:
#             if len(pose_sequence) >= min_frame_threshlod:
#                 if all_same(pose_sequence):
#                     cv2.putText(frame, str(predicted_pose), (50, 50) ,
#                                 cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2)