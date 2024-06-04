To utilize WiSe-FT to train a CLIP model refer to the original repository https://github.com/mlfoundations/wise-ft

We created some custom templates to be added to the src/templates folder and the __init__ file should be edited to add:

```
from .bird_template import bird_template
from .food_template import food_template
from .Aircrafts import plane_template
from .competition_template import competition_template

```

