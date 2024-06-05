This scripts allow to train a ResNet50 + KAN on different dataset. To speed the training up the model starts from the default weight, that are pre-trained on imageNet. The default architecture is also modified to account for the different number of classes in the datasets.

To run the code, it is sufficient to change the path of the dataset, if it does not come from Torchvision.

The requirements.txt file contains the information about the environment used for training.
