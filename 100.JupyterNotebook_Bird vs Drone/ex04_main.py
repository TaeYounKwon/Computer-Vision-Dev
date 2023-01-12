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
    ### 0. Augmentation (train & valid)
    train_aug = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.Resize(width= 200, height= 200),
        # A.RandomCrop(width= 180, height= 180),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06,
                                    rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        A.RandomBrightnessContrast(p= 0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    valid_aug = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.Resize(width= 200, height= 200),
        # A.CenterCrop(width= 180, height= 180),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    ### 1. Loading Classification Dataset
    train_dataset = CustomDataset("./0110/dataset/train" , transform= train_aug)
    valid_dataset = CustomDataset("./0110/dataset/val"   , transform= valid_aug)
    test_dataset  = CustomDataset("./0110/dataset/test"  , transform= valid_aug)

    
    ### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size= 128, shuffle= True , num_workers= 2, pin_memory= True)
    valid_loader = DataLoader(valid_dataset, batch_size= 128, shuffle= False, num_workers= 2, pin_memory= True)
    test_loader  = DataLoader(test_dataset , batch_size= 1, shuffle= False  , num_workers= 2, pin_memory= True)

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

    visualize_augmentation(train_dataset)
    # visualize_augmentation(valid_dataset)
    # visualize_augmentation(test_dataset)

    ### 3. model build
    model_list = []
    
    # model1 = models.swin_t(weights='IMAGENET1K_V1')
    # model1.head = nn.Linear(in_features=768, out_features=3)
    # model1.to(device)

    model2 = models.__dict__["resnet50"](pretrained= True)
    model2.fc = nn.Linear(in_features= 2048, out_features= 3)
    model2.to(device)

    # model3 = models.__dict__["resnet18"](pretrained= True)
    # model3.fc = nn.Linear(in_features= 512, out_features= 3)
    # model3.to(device)

    # model4 = models.__dict__["vgg16"](pretrained= True)
    # model4.classifier[6] = nn.Linear(in_features= 4096, out_features= 3)
    # model4.to(device)

    # model5 = models.__dict__["shufflenet_v2_x0_5"](pretrained= True)
    # model5.fc = nn.Linear(in_features= 4096, out_features= 3)
    # model5.to(device)

    # model_list= [model1, model2, model3, model4, model5]
    model_list= [model2]

    #### 4 epoch, optim loss
    epochs = 10
    loss_function = LabelSmoothingCrossEntropy()

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(valid_loader)
    
    for index, model in enumerate(model_list):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        save_path = f'./best.pt'
        dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=['Epoch', 'Accuracy', 'Loss'])

        if os.path.exists(save_path) :
            best_val_acc = max(pd.read_csv(f'./modelAccuracy.csv')['Accuracy'].tolist())
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

            dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
            dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
            dfForAccuracy.loc[epoch, 'Loss'] = runing_loss

            print(f"epoch [{epoch+1}/{epochs}]"
                f" train loss : {(runing_loss / train_steps):.3f} "
                f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}"
            )

            if val_accuracy > best_val_acc :
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), save_path)

            if epoch == epochs - 1 :
                dfForAccuracy.to_csv(f"./modelAccuracy.csv" , index=False)

if __name__ == '__main__':
    main()



