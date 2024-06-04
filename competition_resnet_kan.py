from http.client import responses
import requests
import json
import os
import torch
import torchvision
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from efficient_kan import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

transform = transforms.Compose(
    [ lambda img: transforms.functional.resize(img, (256,256)),
      transforms.ToTensor()
      ]
)

trainset = torchvision.datasets.ImageFolder(root='/home/disi/test1/competition_data/train',
                                           transform=transform)  ###Questo è il dataloader, basta mettere la cartella di train
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#valset = torchvision.datasets.ImageFolder(root='/home/disi/competition_data/train',
#                                          transform=transform)
#valloader = DataLoader(valset, batch_size=64, shuffle=False)

class_names = trainset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####
"""
resnet50 = models.resnet50(weights='DEFAULT')
num_features = resnet50.fc.in_features
resnet50.fc = KAN([num_features, 256, len(class_names)])
resnet50.to(device)
optimizer = optim.AdamW(resnet50.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.fc.parameters():
    param.requires_grad = True
"""
resnet50 = torch.load("resnet50.pt")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(resnet50.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
loss_prev = 10

for epoch in range(5):
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

    # Update learning rate
    scheduler.step()
####

torch.save(resnet50, "resnet50block.pt")
preds = {}
directory = '/home/disi/test1/competition_data/test'  # La cartella con le immagini di test

#valset = torchvision.datasets.ImageFolder(root='/home/disi/test1/competition_data/test',
#                                          transform=transform)
#valloader = DataLoader(valset, batch_size=64, shuffle=False)
#resnet50 = torch.load("resnet50.pt")
#resnet50.to(device)
resnet50.eval()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_path = os.path.join(directory, filename)
    image = Image.open(image_path)
    image = transform(image)
    image = image.to(device)
    image = image.reshape(1, 3, 256, 256)
    output = resnet50(image)
    result = int(torch.argmax(output))
    preds[filename] = class_names[result].split("_")[0]

res = {
    "images": preds,
    "groupname": "Rick_Astley_Fanclub"
}

print(res)
submit(res)