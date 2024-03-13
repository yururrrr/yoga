# -*- coding: utf-8 -*-
import torch
from model import MattingNetwork
import time
# from torch.utils.data import Dataset
# import os
model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter
from models import CNN_LSTM_frame
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
labels=['downward facing dog', 'four-limbed staff pose', 'half standing forward bend', 'mountain pose', 'plank', 'raised hands pose', 'standing forward bend', 'upward facing dog']


def add_text_to_frame(frame, text, position):
    # Convert tensor to PIL image
    frame = frame[0]
    pil_image = TF.to_pil_image(frame)

    # Draw text on the PIL image
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 16)

    draw.text(position, text, fill=(0, 0, 0), font=font)  # You can adjust the position and color as needed

    # Convert the modified PIL image back to a tensor
    modified_frame = TF.to_tensor(pil_image)
    modified_frame = modified_frame.to(frame.device, frame.dtype)
    
    # Add back the first dimension
    modified_frame = torch.unsqueeze(modified_frame, 0)

    return modified_frame
def all_same(sequence):
    return all(x == sequence[0] for x in sequence)

CNN_model = CNN_LSTM_frame(num_classes=8, cnn_hidden_size=512, lstm_hidden_size=256)
CNN_model.load_state_dict(torch.load('model/CNN_LSTM_mat.pt'))
CNN_model.eval()
CNN_model.cuda()
min_frame_threshlod = 30


# video_name = ['test3_mat']
home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
video_path = home_path+'VIDEO/test3_mat.mp4'

# for i in VideoDataset(root_dir=home_path, transform=None):
# videos = ['1_front', '1_left_front', '1_side', '2_front', '2_left_front', '2_side', '3_front', '3_left_front', '3_side', '4_front', '4_left_front', '4_right_front',
#           '5_front', '5_left_front', '5_right_front', '6_side', '7_side', '8_side']
videos = ['1_left_front']
start_time = time.time()

for i in range(len(videos)):
    reader = VideoReader('C:/Users/VIPLAB/Desktop/yuru/VIDEO/'+videos[i]+'.mp4', transform=ToTensor())
    writer = VideoWriter(home_path+'output/predicted_'+videos[i]+'_mat.mp4', frame_rate=30)

    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
    rec = [None] * 4                                       # Initial recurrent states.
    downsample_ratio = 0.25                                # Adjust based on your video.
    frame_count = 0
    
    pose_sequence = []
    with torch.no_grad():
        for src in DataLoader(reader):     
            # print(len(src))
            # RGB tensor normalized to 0 ~ 1.
            fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
            com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
            
            output = CNN_model(com.cuda())
            _, predicted = torch.max(output, 1)
            predicted_pose = predicted.item()
            
            pose_sequence.append(predicted_pose)
            if len(pose_sequence) >= min_frame_threshlod:
                if all_same(pose_sequence):
                    com_text = add_text_to_frame(com, labels[predicted_pose], (100, 50))
                    writer.write(com_text)
                    pose_sequence = []
            else:
                com_text = add_text_to_frame(com, labels[predicted_pose], (100, 500))
                writer.write(com_text)
                # writer.write(com)                              # Write frame.

            frame_count +=1
            # if frame_count>=4000: break

end_time = time.time()
execution_time = end_time-start_time
print('Execution time:', execution_time, ' s')