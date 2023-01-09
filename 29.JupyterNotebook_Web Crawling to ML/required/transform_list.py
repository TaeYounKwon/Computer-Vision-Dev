import albumentations as A
from albumentations.pytorch import ToTensorV2

# train augmentation
transform_train0 = A.Compose([
    A.Resize(height= 256, width= 256),
    ToTensorV2()
])

transform_train1 = [
    A.Resize(height=256, width=256),
    A.RandomCrop(height= 224, width= 224), # 랜덤 crop 224 x 224
    A.HorizontalFlip(p= 0.5),              # 좌우
    ToTensorV2()
]

transform_train2 = [
    A.Resize(height=256, width=256),
    A.RandomCrop(height= 224, width= 224), # 랜덤 crop 224 x 224
    A.OneOf([
        A.HorizontalFlip(p = 1),    # 좌우 반전
        A.VerticalFlip(p = 1),      # 상하 반전
        A.RandomRotate90(p = 1),    # 회전
    ], p= 1),
    ToTensorV2()
]

transform_train3 = [
    A.Resize(height=256, width=256),
    A.RandomCrop(height= 224, width= 224), # 랜덤 crop 224 x 224
    A.OneOf([
        A.MotionBlur(p=1),          # 모션 블러 필터
        A.OpticalDistortion(p=1),   # 광학 외곡
        A.GaussNoise(p=1),          # 가우시안 블러 필터
    ], p= 1),
    ToTensorV2()
]

transform_train4 = [
    A.Resize(height=256, width=256),
    A.RandomCrop(height= 224, width= 224), # 랜덤 crop 224 x 224
    A.OneOf([
        A.HorizontalFlip(p = 1),    # 좌우 반전
        A.VerticalFlip(p = 1),      # 상하 반전
        A.RandomRotate90(p = 1),    # 회전
    ], p= 1),
    A.OneOf([
        A.MotionBlur(p=1),          # 모션 블러 필터
        A.OpticalDistortion(p=1),   # 광학 외곡
        A.GaussNoise(p=1),          # 가우시안 블러 필터
    ], p= 1),
    ToTensorV2()
]

num_of_train_transform = 5
transform_trains = ["transform_train"+str(index) \
    for index in range(num_of_train_transform)]
# ['transform_train0', 'transform_train1', 'transform_train2', 
# 'transform_train3', 'transform_train4']

transform_val = [
    A.Resize(height=256, width=256),
    ToTensorV2()
]
