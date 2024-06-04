from http.client import responses
import requests
import json
import os
import torch
import torchvision
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from efficient_kan import KAN
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import ConvNextV2ForImageClassification

# Submission function
def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# DataLoader
transform = transforms.Compose(
    [ lambda img: transforms.functional.resize(img, (224,224)),
      transforms.ToTensor()
      ]
)

trainset = torchvision.datasets.ImageFolder(root='/home/disi/test1/competition_data/train',
                                           transform=transform)  ###Questo è il dataloader, basta mettere la cartella di train
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

class_names = trainset.classes

# Model
model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224",
                                                         return_dict=False)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))

# Uncomment if you want to use an already saved model
#model = torch.load("convnextv2base.pt")
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)[0]
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
    scheduler.step()
    # Validation

    torch.save(model, "convnextv2base.pt")

    with torch.no_grad():
        #model = torch.load("convnextv2.pt")
        preds = {}
        directory = '/home/disi/test1/competition_data/test'  # La cartella con le immagini di test

        model.eval()
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image = transform(image)
            image = image.to(device)
            image = image.reshape(1, 3, 224, 224)
            output = model(image)[0]
            result = int(torch.argmax(output))
            preds[filename] = class_names[result].split("_")[0]

        res = {
            "images": preds,
            "groupname": "Rick_Astley_Fanclub"
        }

        print(res)
        submit(res)
