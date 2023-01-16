import os
import sys
import torch    
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from ex04_customdataset import CustomDataset
import pandas as pd
from tqdm import tqdm
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pt_파일명 설정
model_try = 1
model_names = 'resnet50'

# 하이퍼 파라미터 설정
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 128
LOSS_FUNCTION = LabelSmoothingCrossEntropy()
HALF_PERCENT = 0.5

######################
def main():    
    ### 0. Augmentation (train & valid)
    train_aug = A.Compose([
        # 100퍼(val도 동일 적용)
        A.RandomCrop(width= 200, height= 200),
        # 50퍼
        A.RandomRotate90(p=HALF_PERCENT),
        A.VerticalFlip(p=HALF_PERCENT),
        A.HorizontalFlip(p=HALF_PERCENT),
        A.RandomBrightness(limit=0.2, p=HALF_PERCENT),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p = HALF_PERCENT),
        # 30퍼(예시에 p값을 0.3주는게 많아서 0.3으로 설정)
        A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06, rotate_limit=20, p= 0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    valid_aug = A.Compose([
        A.CenterCrop(width= 200, height= 200),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    ### 1. Loading Classification Dataset
    train_dataset = CustomDataset("./dataset/train" , transform= train_aug)
    valid_dataset = CustomDataset("./dataset/val"   , transform= valid_aug)

    ### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True , num_workers= 2, pin_memory= True)
    valid_loader = DataLoader(valid_dataset, batch_size= BATCH_SIZE, shuffle= False, num_workers= 2, pin_memory= True)

    ## visual augmentation
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

    visualize_augmentation(train_dataset)
    visualize_augmentation(train_dataset)
    visualize_augmentation(train_dataset)
    visualize_augmentation(train_dataset)
    visualize_augmentation(train_dataset)
    visualize_augmentation(train_dataset)
    # visualize_augmentation(valid_dataset)
    # visualize_augmentation(test_dataset)
    
    ### 3. model build

    
    # model1 = models.swin_t(weights='IMAGENET1K_V1')
    # model1.head = nn.Linear(in_features=768, out_features=3)
    # model1.to(device)

    # model2 = torch.hub.load('facebookresearch/deit:main',
    #                    'deit_tiny_patch16_224', pretrained=True)

    # model2.fc = nn.Linear(in_features=192, out_features=2)
    # model2.to(device)

    # model3 = models.__dict__["resnet18"](pretrained= True)
    # model3.fc = nn.Linear(in_features= 512, out_features= 3)
    # model3.to(device)


    # model5 = models.__dict__["shufflenet_v2_x0_5"](pretrained= True)
    # model5.fc = nn.Linear(in_features= 4096, out_features= 3)
    # model5.to(device)

    model_list = []

    model = models.__dict__["resnet50"](pretrained= True)
    model.fc = nn.Linear(in_features = 2048, out_features = 6)
    model.to(device)

    model_list= [model]

    #### 4 epoch, optim loss
    epochs = EPOCHS
    loss_function = LOSS_FUNCTION

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(valid_loader)

    ############ 수정하기 ####
    for index, model in enumerate(model_list):
        optimizer = torch.optim.AdamW(model.parameters(), lr= LEARNING_RATE)
        # optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)

        save_path = f'./best_{model_try}.pt'
        dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=['Epoch', 'Accuracy', 'Loss'])

        if os.path.exists(save_path) :
            best_val_acc = max(pd.read_csv(f'./{model_names}_{model_try}.csv')['Accuracy'].tolist())
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
                dfForAccuracy.to_csv(f"./{model_names}_{model_try}.csv" , index=False)

if __name__ == '__main__':
    main()