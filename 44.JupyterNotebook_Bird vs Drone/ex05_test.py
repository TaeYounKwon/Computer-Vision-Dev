import os
import sys
import torch    
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from ex03_customdataset import CustomDataset
import pandas as pd
from tqdm import tqdm
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    test_aug = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.Resize(width= 200, height= 200),
        # A.CenterCrop(width= 180, height= 180),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dataset  = CustomDataset("./dataset/test" , transform= test_aug)
    test_loader = DataLoader(test_dataset, batch_size= 1, shuffle= False, num_workers= 2, pin_memory= True)


    ###### 수정해야 할 부분 !!!!!!!!!!!!!!!!!!!

    # model = models.__dict__["resnet50"](pretrained=False)
    model = models.__dict__["resnet50"](pretrained= False)
    model.fc = nn.Linear(in_features= 2048, out_features= 3)

    model.load_state_dict(torch.load("./best.pt", map_location=device))
    model.to(device)

    criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    test(model, test_loader, device)

def acc_function(correct, total) :
    acc = correct / total * 100
    return acc

def test(model, data_loader, device) :
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader) :
            images, labels = image.to(device), label.to(device)
            output = model(images)
            _, argmax = torch.max(output, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()
        acc = acc_function(correct, total)
        print(f"acc >> {acc}%" )

if __name__ == '__main__':
    main()