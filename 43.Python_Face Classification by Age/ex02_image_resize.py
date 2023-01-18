from PIL import Image # pip install Pillow
import os
import glob

def padding(pil_img, set_size, background_color):
    width, height = pil_img.size
    w = set_size - width
    h = set_size - height
    if height != set_size & width != set_size :
        result = Image.new(pil_img.mode, (set_size, set_size), background_color)
        # image add(추가할 이미지, 붙일 위치(가로, 세로))
        result.paste(pil_img, (h // 2, w // 2))
        return result
    else:
        return pil_img

file_path = "./raw_data"   # 위치 자신의 파일경로에 맞게 변경 필요
image_list = glob.glob(os.path.join(file_path, "*","*"))
folder_list = glob.glob(os.path.join(file_path, "*"))
for create_folder in folder_list:
    create_folder= create_folder.replace("raw_data", "resize_data") 
    os.makedirs(create_folder, exist_ok=True) 

for item_path in image_list:
    try:
        newfile_path = item_path.replace("raw_data", "resize_data") # raw_data -> resize_data
        # 확장자 변환 (raw_data 확장자에 따라 수정 필요)
        newfile_path = newfile_path.replace(".jpg", ".png")         # png로 변환
        # newfile_path = newfile_path.replace(".jpeg", ".png")
        # newfile_path = newfile_path.replace(".PNG", ".png")
        # newfile_path = newfile_path.replace(".JPG", ".png")
        
        image = Image.open(item_path)  
        new_image = image.resize((112,112))
        new_image = padding(new_image, 224, (0,0,0)) 
        new_image.save(newfile_path)
    except Exception as e:    
        print('Error 발생', e)