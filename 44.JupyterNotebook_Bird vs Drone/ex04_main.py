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
from timm.loss import BinaryCrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
HALF_PERCENT = 0.5
FULL_PERCENT = 1.0
LOSS_FUNCTION = BinaryCrossEntropy()

FIX = 2  ## 새로운 모델 훈련시 고쳐주시면 됩니다
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def main():    
    ### 0. Augmentation (train & valid)
    train_aug = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.RandomCrop(width= 200, height= 200),
        A.HorizontalFlip(p= HALF_PERCENT),
        A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06,
                                    rotate_limit=20, p= HALF_PERCENT),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p = 1),
        
        ## 낮은 확률
        A.RandomBrightnessContrast(p= 0.2),
        A.VerticalFlip(p= 0.2),     # 뒤집어 있을 확률 낮음
        A.ToGray(p= 0.333),

        ## 날씨 상황
        A.OneOf([
            A.RandomFog(fog_coef_lower= 0.3, fog_coef_upper= 0.8,
                        alpha_coef= 0.03, p= FULL_PERCENT),             # 안개
            A.RandomSunFlare(flare_roi= (0, 0, 0.05, 0.001), 
                        angle_lower= 0.1, p= FULL_PERCENT),             # 눈뽕
            A.RandomSnow(brightness_coeff= 2.5, snow_point_lower= 0.1, 
                        snow_point_upper= 0.3, p= FULL_PERCENT),        # 눈 (눈만 0.2)
            A.RandomRain(brightness_coefficient= 0.7, drop_width= 1,    # 비
                        blur_value= 3, p= FULL_PERCENT), 
        ], p = HALF_PERCENT),

        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    valid_aug = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.CenterCrop(width= 200, height= 200),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    ### 1. Loading Classification Dataset
    train_dataset = CustomDataset("./0111/dataset/train" , transform= train_aug)
    valid_dataset = CustomDataset("./0111/dataset/val"   , transform= valid_aug)
    
    ### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True , num_workers= 2, pin_memory= True)
    valid_loader = DataLoader(valid_dataset, batch_size= BATCH_SIZE, shuffle= False, num_workers= 2, pin_memory= True)

    ### visual augmentation
    def visualize_augmentation(dataset, idx = 0, cols= 5):
        dataset = copy.deepcopy(dataset)
        samples = 5
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(
            t, (A.Normalize, ToTensorV2)
        )])
        rows = samples // cols
        figure, ax = plt.subplots(nrows= rows, ncols= cols, figsize=(12,6))

        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        
        plt.tight_layout()
        plt.show()

    # visualize_augmentation(train_dataset)
    # visualize_augmentation(valid_dataset)
    # visualize_augmentation(test_dataset)
    
    ### 3. model build
    model_list = []
    model1 = models.mobilenet_v2(pretrained= True)
    model1.classifier[3] = nn.Linear(in_features=1280, out_features= 2)
    model1.to(device)

    model2 = models.__dict__["resnet50"](pretrained= True)
    model2.fc = nn.Linear(in_features= 2048, out_features= 2)
    model2.to(device)

    # model3 = models.__dict__["resnet18"](pretrained= True)
    # model3.fc = nn.Linear(in_features= 512, out_features= 3)
    # model3.to(device)

    # model4 = models.__dict__["vgg16"](pretrained= True)
    # model4.classifier[6] = nn.Linear(in_features= 4096, out_features= 3)
    # model4.to(device)

    model_list= [model1]

    #### 4 epoch, optim loss
    epochs = EPOCHS
    loss_function = LOSS_FUNCTION

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(valid_loader)
    
    for index, model in enumerate(model_list):
        optimizer = torch.optim.AdamW(model.parameters(), lr= LEARNING_RATE)        
        save_path = f'./0111/best{str(FIX)}.pt'
        dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=['Epoch', 'Accuracy', 'Loss'])

        if os.path.exists(save_path) :
            best_val_acc = max(pd.read_csv(f'./0111/modelAccuracy{str(FIX)}.csv')['Accuracy'].tolist())
            model.load_state_dict(torch.load(save_path))

        for epoch in range(epochs) :
            runing_loss = 0
            val_acc = 0
            train_acc = 0

            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
            for step, data in enumerate(train_bar) :
                images , labels = data
                images , labels = images.to(device) , labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
                loss.backward()
                optimizer.step()
                runing_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch+1} / {epoch}], loss{loss.data:.3f}"

            model.eval()
            with torch.no_grad() :
                valid_bar = tqdm(valid_loader, file=sys.stdout, colour='red')
                for data in valid_bar :
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

            val_accuracy = val_acc / len(valid_dataset)
            train_accuracy = train_acc / len(train_dataset)

            print(f"epoch [{epoch+1}/{epochs}]"
                f" train loss : {(runing_loss / train_steps):.3f} "
                f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}"
            )

            dfForAccuracy.loc[epoch, 'Epoch']    = epoch + 1
            dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 4) * 100
            dfForAccuracy.loc[epoch, 'Loss']     = round( (runing_loss / train_steps), 4)
            
            if val_accuracy > best_val_acc :
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), save_path)

            if epoch == epochs - 1 :
                dfForAccuracy.to_csv(f"./0111/modelAccuracy{str(FIX)}.csv" , index=False)

if __name__ == '__main__':
    main()