This script allows to fully train a ResNet50 on the NABirds dataset. To speed the training up the model starts from the default weight, that are pre-trained on imageNet. The default ResNet50 architecture is also modified to account for the different number of classes in the NABirds dataset.

To run this model the NABirds dataset need to be downloaded in advance, and the 'dataset_path' variable must be changed to the dataset folder.

The `ResNet50_env.yaml` file contains the information about the environment used for training, but any environment with `torch=2.3.0` and `torchvision==0.18.0` should work properly.
