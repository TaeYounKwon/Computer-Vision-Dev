import os 
import glob 

import torch 

from custom_dataset import custom_dataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")