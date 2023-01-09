import os
import glob
from PIL import Image
train_path = "./dataset/train_image"
test_path  = "./dataset/test_image"

# 이미즈 리사이즈 정사각형 만들기
IMG_FORMAT = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG" ]

def expand2square(pil_image, background_color):
    width, height = pil_image.size
    if width == height:
        return pil_image
    elif width > height:
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width - height)//2 ))
        return result
    else:
        result = Image.new(pil_image.mode, (height, height), background_color)
        result.paste(pil_image, ( (height - width )//2 , 0))
        return result

# 폴더에서 파일을 검색
def image_file(image_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(image_folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in IMG_FORMAT:
                root = os.path.join(path, filename)
                all_root.append(root)
            else:
                print("no image file...")
                continue
    return all_root