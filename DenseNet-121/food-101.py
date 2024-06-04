import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

root_dir = './data/Food101'


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



train_dataset = Food101(root=root_dir, split='train', transform=transform, download=True)


test_dataset = Food101(root=root_dir, split='test', transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


for images, labels in train_loader:
    print(images.size(), labels.size())
    break
    


def show_images(images, labels):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(images[i], (1, 2, 0)))  # Permuta le dimensioni dell'immagine per adattarle al formato di pyplot
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')  # Nascondi gli assi
    plt.show()

# Ottieni un batch di immagini dal DataLoader


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm  # Import tqdm at the beginning

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming class_names is defined somewhere


model = models.densenet121(pretrained=True)  # Load pretrained DenseNet121
model.classifier = nn.Linear(model.classifier.in_features, 101)

# Move model to device
model = model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)



model_save_path = 'food101_model.pth'

for epoch in range(10):
    # Train
    model.train()
    train_losses = []
    train_accuracies = []
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
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
    

    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader)
    

    # Save model
    torch.save(model.state_dict(), model_save_path)
    
    # Print epoch statistics
    print(f"Epoch {epoch + 1}/10, Train Loss: {sum(train_losses)/len(train_loader):.4f}, Train Accuracy: {sum(train_accuracies)/len(train_loader):.2%}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

# Caricare il modello salvato
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Assicurati che il modello sia in modalità di valutazione
    


