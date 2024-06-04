import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.templates.utils import append_proper_article
import torchvision.datasets

class Food101:
    def __init__(self, preprocess,
                 location=os.path.expanduser('/home/disi/intro2ml/food-101'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        ################# training #################
        self.train_dataset = torchvision.datasets.Food101(root = location, split = 'train', transform=preprocess, download=True)

        prova = self.train_dataset.classes
        res = [x.split("_")[-1] for x in prova]
        self.train_dataset.classes = res

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        ################# testing #################
#        self.test_dataset = torchvision.datasets.Food101(root = location, split = 'test', transform=preprocess, download=True)
        self.test_dataset = torchvision.datasets.Food101(root = location, split = 'train', transform=preprocess, download=True)

        self.test_dataset.classes = res

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = res

