# _tool.py

import torch
from torch.utils.data import random_split
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
# class to test and train
def to_indices(dataset):
    class_to_indices = {}
    for idx, target in enumerate(dataset.labels):
        if target not in class_to_indices:
            class_to_indices[target] = []
        class_to_indices[target].append(idx)
    
    return class_to_indices


def train_test_split(dataset, ratio):
    class_samples = [[] for _ in range(len(dataset.labels))]
    for sample in dataset.samples:
        class_samples[sample[1]].append(sample)

    train_samples = []
    test_samples = []
    for samples in class_samples:
        train_size = int(ratio * len(samples))
        train, test = random_split(samples, [train_size, len(samples) - train_size])
        train_samples.extend(train)
        test_samples.extend(test)
    # print(test_samples[:10])
    return train_samples, test_samples


def test_model(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_dataloader)
    accuracy = correct / total

    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(avg_loss, accuracy * 100))

def visualize_test_results(model, test_dataloader, save_dir):
    model.eval()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        test_loss = 0.0
        correct_val = 0
        total_val = 0
        criterion = nn.CrossEntropyLoss()
        for idx, (inputs, labels) in enumerate(test_dataloader):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            loss_val = criterion(outputs, labels)
            
            # print(f'input={inputs}, output={outputs}, loss_val={loss_val}')
            
            test_loss += loss_val.item()
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    avg_loss_val = test_loss / len(test_dataloader)
    accuracy_val = correct_val / total_val
    print(f'ave_loss={avg_loss_val}, accuracy_val={accuracy_val}')
            # for i in range(inputs.size(0)):
            #     # Load the original image
            #     # image_path = image_paths[i]
            #     # original_image = Image.open(image_path)
            #     original_image = np.array(inputs[i])

            #     # Get the predicted label
            #     predicted_label = predicted[i].item()

            #     # Plot the original image with predicted label
            #     plt.figure()
            #     plt.imshow(original_image)
            #     plt.title(f'Predicted Label: {predicted_label}')
            #     plt.axis('off')

            #     # Save the image with prediction
            #     save_path = os.path.join(save_dir, f'test_{idx}_{i}.png')
            #     plt.savefig(save_path)
            #     plt.close()
