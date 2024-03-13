import cv2
import numpy as np
import torch
from torchvision.treansforms import transforms
from models import CNN_LSTM_frame


model = CNN_LSTM_frame()
model = torch.load('')
model.eval()
predict = model('./model/CNN_LSTM32_mat.pt')

cap = cv2.VideoCapture('')

frame_count = 0
start_time = None
pose_sequence = []


while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print('Ignore empty camera frame.')
        break
       
    # with torch.no_grad():
    #     for batch