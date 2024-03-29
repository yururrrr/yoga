import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # Flatten the output
        x = x.view(-1, 128 * 56 * 56)

        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        x = nn.functional.softmax(x, dim=1)

        return x
    
class CNN_LSTM_frame(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size, lstm_hidden_size, net):
        super(CNN_LSTM_frame, self).__init__()

        if net=='resnet18':
            self.cnn_model = resnet18(pretrained=True)
        elif net=='resnet34':
            self.cnn_model = resnet34(pretrained=True)
        elif net=='resnet50':
            self.cnn_model = resnet50(pretrained=True)

        
        # delete full-connected layer in CNN
        self.cnn_model.fc = nn.Identity()

        # add LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # delete full-connected layer in CNN
        self.cnn_model.fc = nn.Identity()

        # add LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        cnn_output = self.cnn_model(x)
        
        cnn_output = cnn_output.view(x.size(0), -1, 512)

        lstm_output, _ = self.lstm(cnn_output)

        lstm_output = lstm_output[:, -1, :]

        # Fully connected layers
        x = nn.functional.relu(self.fc1(lstm_output))
        x = self.fc2(x)

        return x

class CNN_LSTM_keypoint(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size, lstm_hidden_size, num_keypoints, num_channels):
        super(CNN_LSTM_keypoint, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_keypoints * num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # add LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):

        cnn_output = self.cnn_model(x)
        
        cnn_output = cnn_output.view(x.size(0), -1, x.size(2))

        lstm_output, _ = self.lstm(cnn_output)

        lstm_output = lstm_output[:, -1, :]

        # Fully connected layers
        # x = nn.functional.relu(self.fc1(lstm_output))
        x = self.fc(lstm_output)

        return x