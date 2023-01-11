import pandas as pd
import numpy as np
import torch

from torch import nn 
from torch import optim
from torch.utils.data import Dataset, DataLoader 

class customDataset(Dataset):
    def __init__(self, fileName):
        df = pd.read_csv(fileName)
        self.length = len(df)
        
        self.x1 = df.iloc[:,0].values
        self.x2 = df.iloc[:,1].values 
        print(self.x1)
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self.length
    
test = customDataset("./data.csv")
for i in test:
    print(i)