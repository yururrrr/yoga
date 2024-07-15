import torch
from torch.utils.data import random_split
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
# class to test and train
def to_indices(dataset):
    class_to_indices = {}
    for idx, target in enumerate(dataset.labels):
        if target not in class_to_indices:
            class_to_indices[target] = []
        class_to_indices[target].append(idx)
    
    return class_to_indices

# def train_test_split(dataset, ratio):
#     class_samples = [[] for _ in range(len(dataset.labels))]
#     for sample in dataset.samples:
#         class_samples[sample[1]].append(sample)

#     train_samples = []
#     test_samples = []
#     for samples in class_samples:
#         train_size = int(ratio * len(samples))
#         train, test = random_split(samples, [train_size, len(samples) - train_size])
#         train_samples.extend(train)
#         test_samples.extend(test)
#     # print(test_samples[:10])
#     return train_samples, test_samples
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
def split_dataset(dataset, test_size=0.2, val_size=0.1):
    indices = list(range(len(dataset)))
    labels = [sample[1] for sample in dataset.samples]
    
    # Split the data into training + validation and test sets
    train_val_indices, test_indices, _, _ = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Further split the training + validation set into training and validation sets
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, [labels[i] for i in train_val_indices], test_size=val_size, stratify=[labels[i] for i in train_val_indices], random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def visualize_test_results(model, test_dataloader, save_dir, device):
    model.eval()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        test_loss = 0.0
        correct_val = 0
        total_val = 0
        criterion = nn.CrossEntropyLoss()
        for idx, (inputs, labels, lenghts) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            lenghts = lenghts.cpu()
            outputs = model(inputs, lenghts)
            _, predicted = outputs.max(1)
            loss_val = criterion(outputs, labels)
            
            test_loss += loss_val.item()
            total_val += labels.size(0)
            correct_val += (predicted==labels).sum().item()
            
            for i in range(inputs.size(0)):
                # print(inputs.size(0))
                for j in range(inputs.size(1)):  # Loop through the sequence length
                    # Convert tensor to numpy array and permute dimensions for visualization
                    # print(inputs[i, j].shape)
                    original_image = np.array(inputs[i, j].permute(1, 2, 0).cpu())

                    # Get the predicted label
                    predicted_label = predicted[i].item()

                    # Plot the original image with predicted label
                    plt.figure()
                    plt.imshow(original_image)
                    plt.title(f'Predicted Label: {predicted_label}')
                    plt.axis('off')

                    # Save the image with prediction
                    save_path = os.path.join(save_dir, f'test_{idx}_{i}_{j}.png')
                    plt.savefig(save_path)
                    plt.close()

    avg_loss_val = test_loss / len(test_dataloader)
    accuracy_val = correct_val / total_val
    print(f'ave_loss={avg_loss_val}, accuracy_val={accuracy_val}')
    
    # if width and x1 is not None:
    #     x2 = x1 + width
    # elif width and x2 is not None:
    #     x1 = x2 - width

    # if height and y1 is not None:
    #     y2 = y1 + height
    # elif height and y2 is not None:
    #     y1 = y2 - height

    # if x_center:
    #     x1, x2 = x_center - width / 2, x_center + width / 2

    # if y_center:
    #     y1, y2 = y_center - height / 2, y_center + height / 2

    # x1 = x1 or 0
    # y1 = y1 or 0
    # x2 = x2 or clip.size[0]
    # y2 = y2 or clip.size[1]

    # return clip.fl_image(lambda pic: pic[int(y1) : int(y2), int(x1) : int(x2)], apply_to=["mask"])