
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from pysrc.data import mnistm
from pysrc.models import small

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
