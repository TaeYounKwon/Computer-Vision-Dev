import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split

def split_data(data_path):
    for each_path in os.listdir(data_path):
        total_path = os.path.join(data_path, each_path)    
        images = glob(os.path.join(total_path, "*.png"))
        
        train_set, val_set = train_test_split(images , test_size= 0.2, random_state= 777)
        val_set, test_set  = train_test_split(val_set, test_size= 0.5, random_state= 777)

        print("\n-------" ,total_path, "-------")
        print("No. of Total Image: ", len(images))
        print("No. of Train Image: ", len(train_set))
        print("No. of Val Image: "  , len(val_set))
        print("No. of Test Image: " , len(test_set))
        
        save_in_folder(train_set, mode = "train")
        save_in_folder(val_set  , mode = "val")
        save_in_folder(test_set , mode = "test")


def save_in_folder(data, mode):
    for path in data:
        new_path = path.replace("resize_data", f"dataset\{mode}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True) 
             
        # 이미지 읽어와서 저장
        image = cv2.imread(path)
        cv2.imwrite(new_path, image)

data_path = "./0110/resize_data" 
split_data(data_path)