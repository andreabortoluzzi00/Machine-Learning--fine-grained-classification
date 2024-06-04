import os
import sys
import pandas as pd
import sklearn
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from torch import nn
import torchvision.models as models
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from efficient_kan import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_image_labels(dataset_path=''):
    labels = {}

    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id

    return labels


def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}

    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path

    return paths


def load_train_test_split(dataset_path=''):
    train_images = []
    test_images = []

    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train:
                train_images.append(image_id)
            else:
                test_images.append(image_id)

    return train_images, test_images


if __name__ == '__main__':
    dataset_path = '/nabirds'
    image_path = 'images'

    # Load in the image data
    # Assumes that the images have been extracted into a directory called "images"
    image_paths = load_image_paths(dataset_path, path_prefix=image_path)
    image_class_labels = load_image_labels(dataset_path)

    # Load in the class data
    class_names = load_class_names(dataset_path)

    # Load in the train / test split
    train_images, test_images = load_train_test_split(dataset_path)

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode


class CustomImageDataset(Dataset):
    def __init__(self, image_class_labels, images, image_paths, dataset_path, test=False):
        self.img_labels = image_class_labels
        self.images = images
        self.image_paths = image_paths
        self.dataset_path = dataset_path
        self.test = test

    def __len__(self):
        # Return the number of samples in the split
        return len(self.images)

    def __getitem__(self, idx):
        if self.test == False:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.CenterCrop(224)
            ])
        else:
            transform = transforms.Compose([transforms.Resize((224,224))])
        img_id = self.images[idx]
        img_path = os.path.join(dataset_path, image_paths[img_id])
        image = read_image(img_path, ImageReadMode.RGB)
        image = transform(image)
        image = image / 255

        label = int(self.img_labels[img_id])

        return image, label

train_data = CustomImageDataset(image_class_labels, train_images, image_paths, dataset_path)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_data = CustomImageDataset(image_class_labels, test_images, image_paths, dataset_path, test=True)
valloader = torch.utils.data.DataLoader(test_data, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = torch.load("nabirdsdensenet.pt")
#num_features = resnet50.fc.in_features
#resnet50.classifier = nn.Linear(resnet50.classifier.in_features, len(class_names))
resnet50.to(device)
optimizer = optim.AdamW(resnet50.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.classifier.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Train
    resnet50.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()
            output = resnet50(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    resnet50.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            output = resnet50(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()
    torch.save(resnet50, "nabirdsdensenet.pt")
    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )