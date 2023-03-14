import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np
from torchvision import transforms

#### 경고문, threshold ###
warning_text = '[[ Warning ]]'
threshold_num = 95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = {0:"bird" , 1:"drone"}

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######## 모델 수정#########
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
model.load_state_dict(torch.load("./0111/12nd.pt",map_location=device))
model = model.to(device)
model.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 850)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)

if cap.isOpened():
  while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    input_img = data_transforms(pil_img).reshape([1,3,200,200]).to(device)
    out = model(input_img)
    softmax_result = F.softmax(out)
    top1_prob, top1_label = torch.topk(softmax_result, 1)

    # CCTV에 정확도(%) 표시 
    acc = str(round(top1_prob.item()*100, 3)) + "%"
    
    # thresholed를 위한 정확도
    acc_num = (round(top1_prob.item()*100, 3))
    
    # thresholed를 위한 레이블 이름
    object_label = labels.get(int(top1_label))

    if (acc_num > threshold_num) & (object_label=='drone') :
        cv2.putText(frame, warning_text, (150, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0),3)
        print("Drone Appears!!!!")   

    # label 이름 표시
    cv2.putText(frame, labels.get(int(top1_label)), (10, 400), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 1)
    
    # 정확도 표시
    cv2.putText(frame, acc, (10, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255 ,255 ), 1)
    print(acc, labels.get(int(top1_label)))

    cv2.imshow("Object Detectation", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    if cv2.waitKey(25) == 27 : 
        break
else:
    print('영상 읽기 실패...')

cap.release()
cv2.destroyAllWindows()









