from http.client import responses
import requests
import json
import os
import torch
import torchvision
from torchvision.io import read_image
from torchvision.io import ImageReadMode

def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

dataset = torchvision.datasets.ImageFolder(root = '/home/disi/intro2ml/nabirds/images', transform=transform) ###Questo è il dataloader, basta mettere la cartella di train

####
#QUA IN MEZZO CI METTETE IL MODELLO
####

preds = {}
directory = '/home/disi/intro2ml/nabirds/images/0295' #La cartella con le immagini di test

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_path = os.path.join(directory, filename)
    
    
    ### mettere qui il modello e fare la previsione es.per CLIP
    image = Image.open(os.path.join(dataset_path, filename))
    image = transform(image).unsqueeze(0)
    output = model.forward(image)
    result = torch.argmax(output)
    ###
    
    
    preds[filename] = int(result)


res = {
    "images": preds,
    "groupname": "your_group_name"
}

submit(res)