# _tool.py

import torch
from torch.utils.data import random_split
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
