import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

def load_image_labels(file_name, dataset_path=''):

    '''
    Creates a dictionary having the image name as the key and its label as the value
    starting from the name of the txt file containing the labels and its relative path.

    The file must be a txt containing on each row a image id and its label, separated by a single space.
    If different the function must be modified.
    '''

    labels = {}
  
    with open(os.path.join(dataset_path, file_name)) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id

def load_image_paths(file_name, dataset_path='', image_folder=''):

    '''
    Creates a dictionary having the image name as the key and the relative path to it as the value
    starting from the name of the txt file containing the images, its relative path and the name of the folder containing the images

    Il file deve essere un txt che contiene l'id dell'immagine separato dal path da uno spazio, uno per riga. 
    Se formattato diversamente la funzione va cambiata

    '''
    
    paths = {}
    
    with open(os.path.join(dataset_path, file_name)) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(image_folder, pieces[1])
            paths[image_id] = path
    
    return paths


def load_images(file_name, dataset_path=''):

    '''
    Return a list of images given a text file containing their names and its relative path.
    Ex. a file containing the id of test/training images

    Il file deve essere un txt contenente un id per riga, se differente questa funzione va cambiata
    '''

    with open(os.path.join(dataset_path, file_name)) as f:

        images = []
        for line in f:
            images.append(line)
        
    return images


class CustomImageDataset(Dataset):

    def __init__(self, img_labels_file, images_file, img_path_file, image_folder, dataset_path, train):

        self.img_labels = load_image_labels(img_labels_file, dataset_path)
        self.images = load_images(images_file, dataset_path)
        self.image_paths = load_image_paths(img_path_file, dataset_path, image_folder)
        self.dataset_path = dataset_path
        if train:
            self.transform = transforms.Compose([ ##Volendo le trasformazioni si possono cambiare
                transforms.Resize((256, 256)),
                transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.CenterCrop(224)
                ])
        else:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), 
                transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        img_id = self.images[idx]
        img_path = os.path.join(self.dataset_path, self.image_paths[img_id])
        image = read_image(img_path, ImageReadMode.RGB )  ##Cambiare qui la modalità da RGB se le immagini sono in formati diversi
        image = self.transform(image)
        image = image
        
        label = int(self.img_labels[img_id])

        return image, label