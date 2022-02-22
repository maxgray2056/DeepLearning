# Deep learning starter project 
These projects are for CPSC8430, Clemson University

## Table of Contents
- [Projects](#projects)

- [Background](#background)
- [Install](#install)
- [Packages](#packages)
- [Contributing](#contributing)

## Projects
- [Project_1](project1/)


## Background

These projects are for CPSC8430, Clemson University

## Install

This project uses Python 3 based on Jupyter Notebook

This project is based on Pytorch 1.7.0 & cudnn 1.10
Official website: <a href="https://pytorch.org/get-started/previous-versions/">Previous PyTorch Versions</a>
```
pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.0+cu110 torchvision==0.8.0 --user
```

## Packages

This list gives all recommended packages (may not necessary)
```sh
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transformtransforms

from torchvision import models
from torchsummary import summary
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import os
import cv2
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
```

## Contributing

This project is contributed by: 
<a href="hao9@g.clemson.edu">hao9@clemson.edu</a>
