import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
      
class NAbirds(Dataset):
    def __init__(self,
                 preprocess,
                 location='/home/disi/intro2ml/nabirds',
                 train = True,
                 classnames=None):
        
        self.dataset_path = location
        print(self.dataset_path)


        class_dict = {}
        classnames = []
  
        with open(os.path.join(self.dataset_path, 'classes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = pieces[0]
                classnames.append(' '.join(pieces[1:]))
                class_dict[class_id] = ' '.join(pieces[1:])
        
        
        self.class_dict = class_dict
        self.classnames = classnames


        labels = {}
    
        with open(os.path.join(self.dataset_path, 'image_class_labels.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                class_id = pieces[1]
                labels[image_id] = class_id
    
        self.labels = labels

        train_images = []
        test_images = []
  
        with open(os.path.join(self.dataset_path, 'train_test_split.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                is_train = int(pieces[1])
                if is_train:
                    train_images.append(image_id)
                else:
                    test_images.append(image_id)

        self.train_images = train_images
        self.test_images = test_images

        paths = {}
    
        with open(os.path.join(self.dataset_path, 'images.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                path = os.path.join('images', pieces[1])
                paths[image_id] = path

    
        paths = {}
        
        with open(os.path.join(self.dataset_path, 'images.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                path = os.path.join('images', pieces[1])
                paths[image_id] = path
        
        self.image_paths = paths
        self.transform = preprocess
        self.train = train

    def __len__(self):
        # Return the number of samples in the split
        if self.train == True:
            return len(self.train_images)
        else:
            return len(self.test_images)
    
    def __getitem__(self, idx):
        if self.train == True:
            img_id = self.train_images[idx]
        else:
            img_id = self.test_images[idx]
        img_path = os.path.join(self.dataset_path, self.image_paths[img_id])
        image = self.transform(PIL.Image.open(img_path))
        
        label = int(self.labels[img_id])

        return image, label
    
class NABirds_CLIP:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('/home/disi/intro2ml/nabirds'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        self.preprocess = preprocess
        self.path = location
        self.train_data = NAbirds(preprocess = self.preprocess, train=True)
        self.test_data = NAbirds(preprocess = self.preprocess, train=False)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, num_workers=num_workers, shuffle=False)

        self.classnames = self.train_data.classnames