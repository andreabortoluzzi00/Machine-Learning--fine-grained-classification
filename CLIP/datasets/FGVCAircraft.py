import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.templates.utils import append_proper_article
import torchvision.datasets

class FGVCAircraft:
    def __init__(self, preprocess,
                 location=os.path.expanduser('/home/disi/intro2ml'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        ################# training #################
        self.train_dataset = torchvision.datasets.FGVCAircraft(root = location, split = 'train', annotation_level='variant', transform=preprocess, download=True)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        ################# testing #################
        self.test_dataset = torchvision.datasets.FGVCAircraft(root = location, split = 'test', annotation_level='variant', transform=preprocess, download=True)


        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.train_dataset.classes