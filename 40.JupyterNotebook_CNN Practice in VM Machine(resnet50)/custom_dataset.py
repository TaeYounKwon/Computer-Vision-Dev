import glob
import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

def createLabelDict(file_path):
    label_dic = {}
    label_list = glob.glob(os.path.join(file_path ,"*"))
    for index, item in enumerate(label_list):
        label_name = item.split("\\")[-1]
        label_dic[label_name] = index
    return label_dic

class CustomDataset(Dataset):
    def __init__(self, file_path, transform= None):
        self.file_path = glob.glob(os.path.join(file_path, "*", "*"))
        self.label_dic = createLabelDict(file_path)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.file_path[index]
        image = cv2.imread(image_path)
        # image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label_name = image_path.split("\\")[3]     
        label_num = int(self.label_dic[label_name])
        label = torch.tensor(label_num)

        return image, label

    def __len__(self):
        return len(self.file_path)