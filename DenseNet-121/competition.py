from http.client import responses
import requests
import json
import os
import torch
import torchvision
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, data=res, headers={"Content-Type": "application/json"})
    try:    
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = torchvision.datasets.ImageFolder(root = 'competition_data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True) ###Questo è il dataloader, basta mettere la cartella di train
#test_dataset = torchvision.datasets.ImageFolder(root = 'competition_data/', transform=train_transform)


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm  # Import tqdm at the beginning

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming class_names is defined somewhere


model = models.densenet121(pretrained=True)  # Load pretrained DenseNet121
model.classifier = nn.Linear(model.classifier.in_features, 100)

# Move model to device
model = model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

start_epoch = 20


for epoch in range(start_epoch):
     #Train
    model.train()
    train_losses = []
    train_accuracies = []
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        accuracy = (output.argmax(dim=1) == labels).float().mean().item()
        train_accuracies.append(accuracy)
    
    
    scheduler.step()

    
    print(f"Epoch {epoch + 1}/{start_epoch}, Train Loss: {sum(train_losses)/len(train_loader):.4f}, Train Accuracy: {sum(train_accuracies)/len(train_loader):.2%}")

torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
}, 'checkpoint_path')


torch.save(model.state_dict(), "prova.pth")

preds = {}
directory = 'competition_data/test'
valset = torchvision.datasets.ImageFolder(root='competition_data/train', transform=transform)
classes = valset.classes

model.load_state_dict(torch.load("prova.pth"))
model.eval()  # Set model to evaluation mode

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_path = os.path.join(directory, filename)
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    result = torch.argmax(output)
    res = classes[result].split("_")[0]
    
    preds[filename] = res

res = {
    "images": preds,
    "groupname": "Rick_Astley_Fanclub"
}

submit(res)
