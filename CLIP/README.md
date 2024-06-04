To utilize WiSe-FT to train a CLIP model refer to the original repository https://github.com/mlfoundations/wise-ft

We created some custom templates to be added to the src/templates folder and the __init__ file should be edited to include:

```
from .bird_template import bird_template
from .food_template import food_template
from .Aircrafts import plane_template
from .competition_template import competition_template
```

Similarly we created some custom dataloaders for all datasets that we used. To employ them they should be added to the src/datasets folder and the __init__ file should be edited to include:

```
from .nabirds import NAbirds, NABirds_CLIP
from .Food101 import Food101
from .competition_dataset import competition_dataset
from .FGVCAircraft import FGVCAircraft
```

NOTE: the 0.8 version of torchvision required to run the wise-ft script does not include both the FGVCAircraft and the Food101 datasets but they can simply be added to the package by including the code https://pytorch.org/vision/main/_modules/torchvision/datasets/fgvc_aircraft.html#FGVCAircraft and https://pytorch.org/vision/main/_modules/torchvision/datasets/food101.html#Food101
