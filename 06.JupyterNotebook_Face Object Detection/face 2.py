import cv2
import numpy as np

# https://github.com/opencv/opencv/tree/master/data/haarcascades 다른cascade

# creating face_cascade and eye_cascade objects
face_cascade = cv2.CascadeClassifier('221208\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('221208\haarcascade_eye.xml')

# 얼굴이미지 가져오기
img = cv2.imread('./face.png')

# print(img.shape)
# cv2.imshow('image show', img)
# cv2.waitKey(0)

# Converting the image into grayscale
gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4) # 4 = 박스4개나오게 하는것


# Defining and drawing the rectangles around the face
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2) 
    # 좌표에대한 데이터, (255, 0, 255) 색상, 2 선의굵기
# cv2.imshow('face', img)
# cv2.waitKey(0)

# 관심영역 만들기
roi_gray = gray[y:(y+h), x:(x+w)]
roi_color = img[y:(y+h), x:(x+w)]

# cv2.imshow('face', img)
# cv2.waitKey(0)

# eyes
eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)  # 바운딩박스안에있는 얼굴에만 gray_scale 줌
print(eyes)
index = 0

# creating for loop in ordder to divide one eye from another
for(ex, ey, ew, eh) in eyes:
    if index == 0 :
        eye_1 = (ex, ey, ew, eh)
    elif index ==1:
        eye_2 = (ex, ey, ew, eh)
    img = cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    index = index + 1

cv2.imshow('face', img)
cv2.waitKey(0)