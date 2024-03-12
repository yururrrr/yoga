from torch.utils.data import Dataset
import os
from PIL import Image
import mediapipe as mp


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(os.listdir(root_dir))
        self.labels_to_index = {label: i for i, label in enumerate(self.labels)}
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []

        for label in self.labels:
            label_path = os.path.join(self.root_dir, label)
            videos = os.listdir(label_path)
            for video in videos:
                video_path = os.path.join(label_path, video)
                frames = os.listdir(video_path)
                for frame in frames:
                    frame_path = os.path.join(video_path, frame)
                    samples.append((frame_path, self.labels_to_index[label]))
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        image = Image.open(frame_path).convert('RGB')

        # mp_pose = mp.solutions.pose
        # pose = mp_pose.Pose()
        # results = pose.process(image)
        # keypoints = []
        # if results.pose_landmarks:
        #     for landmark in results.pose_landmark.landmark:
        #         keypoints.append((landmark.x, landmark.y, landmark.z))

        if self.transform:
            image = self.transform(image)

        return image, label