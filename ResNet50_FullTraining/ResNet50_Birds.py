import os
import sys
import pandas as pd
import sklearn
import numpy as np
import random
import math
from torch import nn
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

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
  
  dataset_path = '/home/disi/intro2ml/nabirds'  #### Change this path to run the model
  image_path  = 'images'
  
  # Load in the image data
  # Assumes that the images have been extracted into a directory called "images"
  image_paths = load_image_paths(dataset_path, path_prefix=image_path)
  image_class_labels = load_image_labels(dataset_path)
  
  # Load in the class data
  class_names = load_class_names(dataset_path)


  # Load in the train / test split
  train_images, test_images = load_train_test_split(dataset_path)

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
                transforms.Resize((256, 256)),
                transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.CenterCrop(224)
                ])
        else:
            transform = transforms.Compose([transforms.Resize((256, 256))],
                                            transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])))
        img_id = self.images[idx]
        img_path = os.path.join(dataset_path, image_paths[img_id])
        image = read_image(img_path, ImageReadMode.RGB )
        image = transform(image)
        image = image/255
        
        label = int(self.img_labels[img_id])

        return image, label

#Load the training data
train_data = CustomImageDataset(image_class_labels, train_images, image_paths, dataset_path)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = torchvision.models.resnet50(weights='DEFAULT')
model.fc = nn.Linear(2048, 1011)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Set hyperparameters
num_epochs = 40
batch_size = 128
learning_rate = 0.001



# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    step = 0
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    torch.save(model, f"resnet_Birds_{epoch+21}iter.pth")
    with open('myfile.txt', 'a') as f:
      f.write(f'Epoch {epoch+21}/{num_epochs}, Loss: {loss.item():.4f}\n')

torch.save(model, "resnet_Birds_40iter.pth")

#Load the test dataset
test_data = CustomImageDataset(image_class_labels, test_images, image_paths, dataset_path, test=True)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)

test_loss, test_acc = 0, 0
loss_fn = nn.CrossEntropyLoss()
model.to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

#Evaluate the model accuracy
with torch.inference_mode(): 
    for X, y in test_loader:
        # Send data to device
        X, y = X.float().to(device), y.to(device)
            
        # 1. Forward pass
        test_pred = model(X)
        # 2. Calculate loss and accuracy
        test_loss += loss_fn(test_pred, y)
        test_acc += accuracy_fn(y_true=y,
            y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
    # Adjust metrics and print out
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
