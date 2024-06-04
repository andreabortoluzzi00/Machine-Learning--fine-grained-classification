This script allows to fully train a ResNet50 on the NABirds dataset. To speed the training up the model starts from the default weight, that are pre-trained on imageNet. The default ResNet50 architecture is also modified to account for the different number of classes in the NABirds dataset.

To run this model it is sufficient to change the 'dataset_path' variable to the dataset folder.
