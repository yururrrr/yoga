import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from models import CNN_LSTM_frame
from dataset import VideoDataset
from utils._tool import  to_indices, visualize_test_results
import time
# import mediapipe as mp

height, width = 112, 112
batch_size = 32
num_classes = 8
cnn_hidden_size = 512
lstm_hidden_size = 256
home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
data_path = home_path + 'yoga-pose-classification/dataset_crop'

torch.manual_seed(42)
labels=['downward facing dog', 'four-limbed staff pose', 'half standing forward bend', 'mountain pose', 'plank', 'raised hands pose', 'standing forward bend', 'upward facing dog']
data_transform = transforms.Compose([
    transforms.Resize((height, width)),     #Resize
    transforms.ToTensor(),                  #Convert to Pytorch tensors
])

dataset = VideoDataset(root_dir=data_path, transform=data_transform)
class_to_indicies = to_indices(dataset)


split_ratio = 0.8  #80% for training, 20% for valid
train_size = int(len(dataset) * split_ratio)
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_size = int(len(train_dataset) * 0.9)
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])


model = CNN_LSTM_frame(num_classes=8, cnn_hidden_size=512, lstm_hidden_size=256, net='resnet50')
model.load_state_dict(torch.load('model/resnet50_LSTM_batch64_crop.pt'))
# model.eval()


test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
save_dir = home_path+'test_results'

# Test the model and save results
visualize_test_results(model, test_dataloader, save_dir)

# test rvm
def test_rvm():
    
    video_name = ['1_left_front']
    home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
    video_path = home_path+'RVM_output/1_left_front.mp4'

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    start_time = None
    pose_sequence = []
    min_time_threshold = 10
    min_frame_threshlod = 30

    def all_same(sequence):
        return all(x == sequence[0] for x in sequence)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{home_path}/output/{video_name[0]}_test.mp4', fourcc, 30, (width, height))

    start = time.time()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('Ignore empty camera frame.')
            break
        tensor_frame = transforms.ToTensor()(frame).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor_frame)
            _, predict = output.max(1)
            predicted_pose = predict.item()
            
        # frame_count += 1
        # pose_sequence.append(predicted_pose)
        
        # if start_time is None:
        #     start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        # else:
        #     current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        #     time_difference = current_time - start_time 
        #     if time_difference >= min_time_threshold:
        #         if len(pose_sequence) >= min_frame_threshlod:
        #             if all_same(pose_sequence):
        cv2.putText(frame, str(labels[predicted_pose]), (50, 50) ,
            cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2)
        out.write(frame)
        
    out.release()

    end = time.time()
    execution = end-start
    print('Execution time:',execution, 's')
    