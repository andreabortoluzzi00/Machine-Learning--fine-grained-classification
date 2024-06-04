import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms




transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


root = './data'


train_dataset = datasets.FGVCAircraft(root=root, split='train', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


test_dataset = datasets.FGVCAircraft(root=root, split='test', transform= transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for images, labels in train_loader:
    print(images.size(), labels)
    break

class_names = train_dataset.classes

import matplotlib.pyplot as plt
import numpy as np


def show_images(images, labels):
   
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        img = images[i] / 2 + 0.5
        npimg = img.numpy()
        axs[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis('off')
    plt.show()

images, labels = next(iter(train_loader))

show_images(images, labels)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.densenet121(pretrained=True)  # Load pretrained DenseNet121
model.classifier = nn.Linear(model.classifier.in_features, )



# Move model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)


for epoch in range(15):
    # Train
    model.train()
    train_losses = []
    train_accuracies = []
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{20}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        accuracy = (output.argmax(dim=1) == labels).float().mean().item()
        train_accuracies.append(accuracy)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    
    # Calculate average validation loss and accuracy
    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader)
    
  

    # Print epoch statistics
    print(f"Epoch {epoch + 1}/{20}, Train Loss: {sum(train_losses)/len(train_loader):.4f}, Train Accuracy: {sum(train_accuracies)/len(train_loader):.2%}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")


