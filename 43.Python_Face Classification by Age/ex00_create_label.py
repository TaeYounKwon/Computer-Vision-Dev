import os 
import glob
import cv2
TWENTY, THIRTY, FOURTY = 20, 30, 40 # 나이 값
labels = ['20F','20M','30F','30M','40F','40M']  # 새로 만들 라벨
MALE, FEMALE = '111', '112'
CUT = 200

# 라벨 폴더 만들기
prev_path = "./previous_data"  # 경로 변경 필요 !!!
raw_path  = "./raw_data"        # raw_data는 그대로 해주시고 raw_data 전 경로 변경 필요 !!!
os.makedirs(raw_path, exist_ok= True)

# raw_data/20F ~ 60M 데이터 만들기
for item in labels:
    os.makedirs(os.path.join(raw_path,item), exist_ok= True)

# previous_data의 폴더 리스트 (20~75)
folder_list = os.listdir(os.path.join(prev_path))
switch_int_age = [int(t) for t in folder_list]  # str(15) -> int(15) 정수화
# [20, 21, ...., 75]

# 나이 15 ~ 75 까지
for age in switch_int_age:
    ##### 20대
    if (age >= TWENTY) & (age < THIRTY):
        ### 20대 여자      
        # 20폴더의 모든 여자 이미지(112), 용량 큰걸로 앞에서 1000장만 사용
        image_list = sorted(glob.glob(os.path.join(prev_path,str(age), FEMALE, '*.jpg')), key=os.path.getsize, reverse=True)[:CUT]  
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png')  # 100006-0.jpg -> 100006-0.png
                new_path = os.path.join(raw_path, labels[0], image_name) # 저장할 새공간   
                image = cv2.imread(image_path)   # 불러오기
                cv2.imwrite(new_path, image)     # 이미지 저장
        except Exception as e :
            print(e)

        ### 20대 남자   
        # 20폴더의 모든 남자 이미지(111), 용량 큰걸로 앞에서 1000장만 사용
        image_list = sorted(glob.glob(os.path.join(prev_path,str(age), MALE, '*.jpg')), key=os.path.getsize, reverse=True)[:CUT] 
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png') 
                new_path = os.path.join(raw_path, labels[1], image_name) 
                image = cv2.imread(image_path)   
                cv2.imwrite(new_path, image)     
        except Exception as e :
            print(e)

    ##### 30대
    elif (age >= THIRTY) & (age < FOURTY):
        ### 30폴더의 모든 여자 이미지(112), 용량 큰걸로 앞에서 1000장만 사용 
        image_list = sorted(glob.glob(os.path.join(prev_path,str(age), FEMALE, '*.jpg')), key=os.path.getsize, reverse=True)[:CUT] 
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png') 
                new_path = os.path.join(raw_path, labels[2], image_name) 
                image = cv2.imread(image_path)   
                cv2.imwrite(new_path, image)    
        except Exception as e :
            print(e)

        ### 30폴더의 모든 남자 이미지(111), 용량 큰걸로 앞에서 1000장만 사용
        image_list = sorted(glob.glob(os.path.join(prev_path,str(age), MALE, '*.jpg')), key=os.path.getsize, reverse=True)[:CUT] 
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png')  
                new_path = os.path.join(raw_path, labels[3], image_name) 
                image = cv2.imread(image_path)  
                cv2.imwrite(new_path, image)     
        except Exception as e :
            print(e)

    ##### 40대
    elif (age >= FOURTY):
        ### 40대 여자      
        image_list = glob.glob(os.path.join(prev_path,str(age),'112', '*.jpg'))
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png') 
                new_path = os.path.join(raw_path, labels[4], image_name)  
                image = cv2.imread(image_path)   
                cv2.imwrite(new_path, image)     
        except Exception as e :
            print(e)

        ### 40대 남자
        image_list = glob.glob(os.path.join(prev_path,str(age),'111', '*.jpg'))
        try:
            for image_path in image_list:
                image_name = image_path.split('\\')[-1].replace(".jpg", '.png')  
                new_path = os.path.join(raw_path, labels[5], image_name) 
                image = cv2.imread(image_path)   
                cv2.imwrite(new_path, image)    
        except Exception as e :
            print(e)