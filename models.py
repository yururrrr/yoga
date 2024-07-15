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
    
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNN_Transformer(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size, transformer_hidden_size, nhead, num_encoder_layers, net):
        super(CNN_Transformer, self).__init__()

        if net == 'resnet18':
            self.cnn_model = resnet18(pretrained=True)
        elif net == 'resnet34':
            self.cnn_model = resnet34(pretrained=True)
        elif net == 'resnet50':
            self.cnn_model = resnet50(weights='IMAGENET1K_V1')

        # 删除全连接层并增加卷积层
        self.cnn_model.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.cnn_model.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.cnn_model.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.cnn_model.fc = nn.Conv2d(2048, cnn_hidden_size, kernel_size=1)

        # Transformer Encoder部分
        self.pos_encoder = PositionalEncoding(cnn_hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=cnn_hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(cnn_hidden_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_output = self.cnn_model(x).view(batch_size, seq_len, -1)
        cnn_output = cnn_output.permute(1, 0, 2)  # Transformer 输入要求的格式 (seq_len, batch_size, feature)
        transformer_output = self.transformer_encoder(self.pos_encoder(cnn_output))
        transformer_output = transformer_output.permute(1, 0, 2)
        transformer_output = transformer_output[:, -1, :]  # 使用最后一个时间步的输出

        x = F.relu(self.fc1(transformer_output))
        x = self.fc2(x)
        return x

# # 测试模型
# model = CNN_Transformer(num_classes=100, cnn_hidden_size=2048, transformer_hidden_size=512, nhead=8, num_encoder_layers=6, net='resnet50')
# input_data = torch.randn(32, 289, 3, 112, 112)  # 示例输入
# output = model(input_data)
# print(output.shape)  # 应输出 (32, 100)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class CNN_LSTM_clip(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size, lstm_hidden_size, net):
        super(CNN_LSTM_clip, self).__init__()

        if net == 'resnet18':
            self.cnn_model = resnet18(weights='IMAGENET1K_V1')
            cnn_out_size = 512
        elif net == 'resnet34':
            self.cnn_model = resnet34(pretrained=True)
            cnn_out_size = 512
        elif net == 'resnet50':
            self.cnn_model = resnet50(weights='IMAGENET1K_V1')
            cnn_out_size = 2048

        self.features = nn.Sequential(*list(self.cnn_model.children())[:-1])
        self.lstm = nn.LSTM(cnn_out_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        # Remove fully connected layer and use global average pooling
        # self.cnn_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.cnn_model.fc = nn.Identity()  # Remove the final fully connected layer


        # remove fully connected layer and add a CNN layer
        # self.cnn_model.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        # self.cnn_model.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        # self.cnn_model.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        # self.cnn_model.fc = nn.Conv2d(2048, num_classes, kernel_size=1)
        # self.cnn_model.fc = nn.Linear(512 * 7 * 7, 2048)

        # add LSTM and attention layers
        # self.lstm = nn.LSTM(input_size=cnn_out_size, hidden_size=lstm_hidden_size, batch_first=True)
        # self.attention = Attention(hidden_size=lstm_hidden_size)

        # Fully connected layers
        # self.fc1 = nn.Linear(lstm_hidden_size, 512)
        # self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, lengths):
        batch_size, seq_len, c, h, w = x.size()
        # print(f'Input shape: {x.shape}')
        device = x.device
        x = x.view(batch_size * seq_len, c, h, w)
        lengths = lengths.cpu().to(torch.int64)
        # cnn_output = self.cnn_model(x).view(batch_size, seq_len, -1)
        # print(f'CNN output shape: {cnn_output.shape}')
        # lstm_output, _ = self.lstm(cnn_output)
        # print(f'LSTM output shape: {lstm_output.shape}')
        # context_vector = self.attention(lstm_output)
        # x = F.relu(self.fc1(context_vector))
        # x = self.fc2(x)

        # 通過ResNet提取特徵
        x = self.features(x)
        x = x.view(batch_size, seq_len, -1)
        
        # 使用 PackedSequence 進行 masking
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM 層
        lstm_out, _ = self.lstm(x_packed)
        
        # 解包
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lengths = lengths.to(device)
        
        # Global Average Pooling
        out = torch.sum(lstm_out, dim=1) / lengths.unsqueeze(1).float()
        
        # 分類層
        out = self.fc(out)
        return out

class CNN_LSTM_frame(nn.Module):
    def __init__(self, num_classes, cnn_hidden_size, lstm_hidden_size, net):
        super(CNN_LSTM_frame, self).__init__()

        if net=='resnet18':
            self.cnn_model = resnet18(weights='IMAGENET1K_V1')
        elif net=='resnet34':
            self.cnn_model = resnet34(pretrained=True)
        elif net=='resnet50':
            self.cnn_model = resnet50(weights='IMAGENET1K_V1')

        
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