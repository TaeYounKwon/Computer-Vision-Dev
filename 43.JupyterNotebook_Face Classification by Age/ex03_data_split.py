import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split

def split_data(data_path):
    for each_path in os.listdir(data_path):
        total_path = os.path.join(data_path, each_path)    
        images = glob(os.path.join(total_path, "*.png"))
        
        train_set, val_set = train_test_split(images , test_size= 0.1, random_state= 777)
        # val_set, test_set  = train_test_split(val_set, test_size= 0.5, random_state= 777)
        
        save_in_folder(train_set, mode = "train")
        save_in_folder(val_set  , mode = "val")
        # save_in_folder(test_set , mode = "test")

def save_in_folder(data, mode):
    for path in data:
        new_path = path.replace("resize_data", f"dataset\{mode}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True) 
        try:
            image = cv2.imread(path)
            cv2.imwrite(new_path, image)
        except Exception as e :
            print(e)

# 본인 경로 수정 ####
data_path = "./resize_data" 
split_data(data_path)