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
            # print('size=', inputs.size())

    
            for i in range(inputs.size(0)):
                # Load the original image
                # image_path = image_paths[i]
                # original_image = Image.open(image_path)
                original_image = np.array(inputs[i].permute(1, 2, 0))

                # Get the predicted label
                predicted_label = predicted[i].item()

                # Plot the original image with predicted label
                plt.figure()
                plt.imshow(original_image)
                plt.title(f'Predicted Label: {predicted_label}')
                # plt.axis('off')

                # # Save the image with prediction
                save_path = os.path.join(save_dir, f'test_{idx}_{i}.png')
                plt.savefig(save_path)
                plt.close()

    avg_loss_val = test_loss / len(test_dataloader)
    accuracy_val = correct_val / total_val
    print(f'ave_loss={avg_loss_val}, accuracy_val={accuracy_val}')

    """
    Returns a new clip in which just a rectangular subregion of the
    original clip is conserved. x1,y1 indicates the top left corner and
    x2,y2 is the lower right corner of the croped region.
    All coordinates are in pixels. Float numbers are accepted.
    
    To crop an arbitrary rectangle:
    
    >>> crop(clip, x1=50, y1=60, x2=460, y2=275)
    
    Only remove the part above y=30:
    
    >>> crop(clip, y1=30)
    
    Crop a rectangle that starts 10 pixels left and is 200px wide
    
    >>> crop(clip, x1=10, width=200)
    
    Crop a rectangle centered in x,y=(300,400), width=50, height=150 :
    
    >>> crop(clip,  x_center=300 , y_center=400,
                        width=50, height=150)
    
    Any combination of the above should work, like for this rectangle
    centered in x=300, with explicit y-boundaries:
    
    >>> crop(x_center=300, width=400, y1=100, y2=600)
    
    """

    if width and x1 is not None:
        x2 = x1 + width
    elif width and x2 is not None:
        x1 = x2 - width

    if height and y1 is not None:
        y2 = y1 + height
    elif height and y2 is not None:
        y1 = y2 - height

    if x_center:
        x1, x2 = x_center - width / 2, x_center + width / 2

    if y_center:
        y1, y2 = y_center - height / 2, y_center + height / 2

    x1 = x1 or 0
    y1 = y1 or 0
    x2 = x2 or clip.size[0]
    y2 = y2 or clip.size[1]

    return clip.fl_image(lambda pic: pic[int(y1) : int(y2), int(x1) : int(x2)], apply_to=["mask"])