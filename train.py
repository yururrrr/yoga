import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoDataset
from models import SimpleCNN, CNN_LSTM_frame

height, width = 112, 112
batch_size = 32
num_classes = 8
cnn_hidden_size = 512
lstm_hidden_size = 256
home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
data_path = home_path + 'yoga-pose-classification/dataset_mat'
model_path = './model/CNN_LSTM32_mat.pt'

torch.manual_seed(42)
labels=['downward facing dog', 'four-limbed staff pose', 'half standing forward bend', 'mountain pose', 'plank', 'raised hands pose', 'standing forward bend', 'upward facing dog']
data_transform = transforms.Compose([
    transforms.Resize((height, width)),     #Resize
    transforms.ToTensor(),                  #Convert to Pytorch tensors
])
 # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=30),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

dataset = VideoDataset(root_dir=data_path, transform=data_transform)

split_ratio = 0.8  #80% for training, 20% for valid
train_size = int(len(dataset) * split_ratio)
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
# Create DataLoader for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the CNN model
num_classes = len(dataset.labels)
print(num_classes, dataset.labels_to_index)
# model = SimpleCNN(num_classes)
# model = CNN_LSTM_frame(num_classes, cnn_hidden_size, lstm_hidden_size)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 50

# writer = SummaryWriter('CNN_LSTM_batch32_mat')

# for epoch in range(num_epochs):
#     # training
#     print('train')
#     model.train()
#     running_loss = 0.0
#     total_train = 0
#     correct_train = 0
    
#     for batch_idx, (inputs, labels) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#         _, predicted = outputs.max(1)
#         total_train += labels.size(0)
#         correct_train += predicted.eq(labels).sum().item()

#         if batch_idx % 10 == 9:  # Log every 10 batches
#             writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_dataloader) + batch_idx)
#             writer.add_scalar('Training Accuracy', correct_train / total_train, epoch * len(train_dataloader) + batch_idx)
#             running_loss = 0.0

#     # Validation loop
#     print('valid')
#     model.eval()
#     validation_loss = 0.0
#     correct_val = 0
#     total_val = 0

#     with torch.no_grad():
#         for batch_idx, (inputs_val, labels_val) in enumerate(valid_dataloader):
#             outputs_val = model(inputs_val)
#             loss_val = criterion(outputs_val, labels_val)
#             validation_loss += loss_val.item()

#             _, predicted_val = outputs_val.max(1)
#             total_val += labels_val.size(0)
#             correct_val += predicted_val.eq(labels_val).sum().item()

#     avg_loss_val = validation_loss / len(valid_dataloader)
#     accuracy_val = correct_val / total_val

#     writer.add_scalar('Validation Loss', avg_loss_val, (epoch+1)*len(train_dataloader))
#     writer.add_scalar('Validation Accuracy', accuracy_val, (epoch+1)*len(train_dataloader))



#     epoch_loss = running_loss / len(train_dataloader)
#     # writer.add_scalar('Loss/train', epoch_loss)
#     # print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}")

#     # Validation loop (similar to previous response)
#     # ...


# torch.save(model.state_dict(), model_path)
# writer.close()

# print("Training finished")