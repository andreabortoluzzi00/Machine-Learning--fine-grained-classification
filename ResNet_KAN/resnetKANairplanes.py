from efficient_kan import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

# DataLoaders
transform = transforms.Compose(
    [ lambda img: transforms.functional.resize(img, (256,256)),
      transforms.ToTensor(),
      transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
      ]
)

trainset = torchvision.datasets.FGVCAircraft(
    root="./data", download=True, transform=transform,
    split="trainval"
)
valset = torchvision.datasets.FGVCAircraft(
    root="./data", download=True, transform=transform,
    split="test"
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

class_names = trainset.classes

# Model definition
resnet50 = models.resnet50(weights='DEFAULT')
num_features = resnet50.fc.in_features
resnet50.fc = KAN([num_features, 256, len(class_names)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)
optimizer = optim.AdamW(resnet50.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.fc.parameters():
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

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
