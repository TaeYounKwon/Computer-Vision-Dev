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

data_transforms = A.Compose([
    A.Resize(width= 200, height= 200),
    A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
    ToTensorV2()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = {0:"Paper" , 1:"Rock", 2:"Scissors"}

model = models.vgg11(pretrained=False)
model.classifier[6] = nn.Linear(in_features=4096, out_features=3)
model.load_state_dict(torch.load("4.pt",map_location=device))
model = model.to(device)
model.eval()

def preprocess(image):
    image = Image.fromarray(image)
    image = np.array(image)
    image = data_transforms(image=image)['image']
    image = image.unsqueeze(0)
    return image

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = labels[prediction]
    return result, score

cap = cv2.VideoCapture(0)  # 0 : 웹캠
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        image_data = preprocess(frame)
        prediction = model(image_data)
        result, _= argmax(prediction)
        print(result)
        cv2.putText(frame, result, (950, 250), cv2.FONT_ITALIC, 3, (0, 255, 0), 1)
        cv2.imshow("Object detectation", frame)
        if cv2.waitKey(25) == 27: # ESC로 off
            break
else:
    print('영상 읽기 실패..')

cap.release()
cv2.destroyAllWindows()


#### 예측 + 확률까지 도출하고 싶은 경우####

# data_transforms = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)

# if cap.isOpened():
#   while True:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(frame)
#     input_img = data_transforms(pil_img).reshape([1,3,224,224]).to(device)
#     out = model(input_img)
#     softmax_result = F.softmax(out)
#     top1_prob, top1_label = torch.topk(softmax_result, 1)
#     cv2.putText(frame, labels.get(int(top1_label)), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)
#     acc = ">> " + str(round(top1_prob.item()*100, 3)) + "%"
#     cv2.putText(frame, acc, (30, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
#     print(acc, labels.get(int(top1_label)))
#     cv2.imshow("Object Detectation", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       
#     if cv2.waitKey(25) ==27 : 
#         break
# else:
#     print('영상 읽기 실패..')

# cap.release()
# cv2.destroyAllWindows()