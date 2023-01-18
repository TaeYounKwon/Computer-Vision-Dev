import numpy as np
import torch
import torch.nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import PIL
import cv2
import torchvision.models as models
import torch.nn as nn

# This is the Label
Labels = {0:"p" , 1:"r", 2:"s"}

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
device = torch.device("cpu")  ##Assigning the Device which will do the calculation
model = models.vgg11(pretrained=False)
model.classifier[6] = nn.Linear(in_features= 4096, out_features= 3)
model.load_state_dict(torch.load("4.pt",map_location=device))
model = model.to(device)  # set where to run the model and matrix calculation
model.eval()  # set the device to eval() mode for testing


# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result, score


def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    print(image)
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.cpu()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Set the webcam
Webcam_720p()

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read()  # Capture each frame

    if fps == 4:
        image = frame[100:450, 150:570]
        image_data = preprocess(image)
        #print(image_data)
        prediction = model(image_data)
        print(prediction)
        result, score= argmax(prediction)
        score=float(score)
        #print(type(score))
        fps = 0
        if score >= 0.5 :
            show_res = result
            show_score = score
        else:
            show_res = "Nothing"
            show_score = score

    fps += 1
    cv2.putText(frame, '%s' % (show_res), (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, '(score = %.5f)' % (show_score), (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
    cv2.imshow("ASL SIGN DETECTER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL SIGN DETECTER")