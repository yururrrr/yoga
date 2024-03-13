import torch
from model import MattingNetwork
from torch.utils.data import Dataset
import os
model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import VideoReader, VideoWriter

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(os.listdir(root_dir))

home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
# for i in VideoDataset(root_dir=home_path, transform=None):
# videos = ['1_front', '1_left_front', '1_side', '2_front', '2_left_front', '2_side', '3_front', '3_left_front', '3_side', '4_front', '4_left_front', '4_right_front',
        #   '5_front', '5_left_front', '5_right_front', '6_side', '7_side', '8_side']
videos = ['test']
for i in range(len(videos)):
    reader = VideoReader('C:/Users/VIPLAB/Desktop/yuru/VIDEO/'+videos[i]+'.mp4', transform=ToTensor())
    writer = VideoWriter('output/'+videos[i]+'.mp4', frame_rate=30)

    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
    rec = [None] * 4                                       # Initial recurrent states.
    downsample_ratio = 0.25                                # Adjust based on your video.

    with torch.no_grad():
        for src in DataLoader(reader):                     # RGB tensor normalized to 0 ~ 1.
            fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
            com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
            writer.write(com)                              # Write frame.