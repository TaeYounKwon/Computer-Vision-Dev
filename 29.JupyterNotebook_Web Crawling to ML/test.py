import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import albumentations as A

from required.transform_list import *
from required.utils import train, validate

import required.hy_parameter
from required.dataset_temp import CustomDataset

# 윈도우 기반 그래픽 카드 엔비디아 사용하고 계신경우
device = torch.device("cpu")

# m1 m2 칩셋 사용하시는분
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

def main():
    # model call
    net = models.__dict__["resnet18"](pretrained=True)  # pretrained = True 
    net.fc = nn.Linear(512,5) # 나는 output 5개
    net.to(device)

    criterion = nn.CrossEntropyLoss() # criterion
    optim = torch.optim.Adam(net.parameters(), lr= required.hy_parameter.lr) # optimizer

    # validation
    val_transform = A.Compose(transform_val)  # val aug
    val_dataset   = CustomDataset("./data/val", # val dataset 
                                transform= val_transform)  
    val_loader   = DataLoader(val_dataset, shuffle=False,  # val dataloader
                            batch_size= required.hy_parameter.batch_size)
    
    # train
    for index, transform_train in enumerate(transform_trains):
        # create train_transform directory
        folder_name_temp = transform_train.split("_")[0] + str(index)  # transform + str(index)
        model_save_dir= os.path.join("./model_save", folder_name_temp)
        os.makedirs(model_save_dir, exist_ok=True)

        train_transform = A.Compose(transform_train) # train aug
        train_dataset = CustomDataset("./data/train", # train dataset
                                transform= train_transform)  
        train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=required.hy_parameter.batch_size)
    
        print(f"\n>>> {folder_name_temp} start :\n")
        train(
            number_epoch = required.hy_parameter.epoch,  # 100 
            train_loader = train_loader,        # 위의 train loader
            val_loader   = val_loader,          # 위의 valid loader
            criterion    = criterion,           # Cross Entropy
            optimizer    = optim,               # Adam
            model        = net,                 # resnet18 + num_class 5로 수정
            save_dir     = model_save_dir,      # ./2022.12/12.22_d57_data/model_save\transform 0~4
            device       = device               # cpu
        )

if __name__ == "__main__":
    main()