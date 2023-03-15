# json xml to yolo
import os
import glob
import copy
import json
import cv2

# label_dict = {
#     "garbage_bag" : 9, 
#     # "sit_board"   : 10, 
#     "street_vendor": 11, 
#     "food_truck": 12, 
#     "banner": 13, 
#     "tent": 14,
#     # "smoke": 15, 
#     "flame": 16, 
#     "pet": 17, 
#     # "fence": 18, 
#     "bench": 19, 
#     "park_pot": 20, 
#     "trash_can": 21, 
#     "rest_area": 22, 
#     "toilet": 23, 
#     # "park_headstone": 24, 
#     "street_lamp": 25, 
#     "park_info": 26, 
# }

def main():
    for use in ['train', 'val']:
        ori_json_path = f"./dataset/{use}/labels"
        new_images_path = f"./datasets/{use}/images"
        new_labels_path = f"./datasets/{use}/labels"
        os.makedirs(new_images_path, exist_ok= True)
        os.makedirs(new_labels_path, exist_ok= True)

        # txt_path = ori_json_path.replace('labels', 'texts')
        json_paths = glob.glob(os.path.join(ori_json_path, "*", "*.json"))

        for json_path in json_paths :
            # 본 데이터셋에서 학습에 필요한 정보만 읽어 반환
            anno_data = read_json(json_path)
            image_path = json_path.replace('labels', 'images')
            image_path = image_path.replace('.json', '.jpg')

            # 이미지 리사이즈
            # image, anno_data = resize(image_path, anno_data, (1470, 810))
            image, anno_data = resize(image_path, anno_data, (960, 540))
            jpg_name, label, yolo_x, yolo_y, yolo_w, yolo_h = json_to_yolo(anno_data)
            
            # save resized image
            cv2.imwrite(os.path.join(new_images_path, jpg_name), image)

            # text file
            txt_name = jpg_name.replace('.jpg', '.txt')
            os.makedirs(new_labels_path, exist_ok= True)

            # text save
            with open(f"{new_labels_path}/{txt_name}", 'a') as f :
                f.write(f"{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")
            print(jpg_name)
            print(txt_name)
            exit()
            
def read_json(json_path):
    # 본 데이터셋에서 학습에 필요한 정보만 읽어 반환
    with open(json_path, 'r', encoding="utf8") as j:
        json_data = json.load(j)

    images = json_data['images']
    annotations = json_data['annotations']

    filename = images['ori_file_name']
    height = images['height']
    width = images['width']

    annos = []
    for annotation in annotations:
        label = annotation['object_class']
        label_num = annotation['object_id']
        bbox = annotation['bbox']
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]

        anno = {
            'label': label,
            'label_num': label_num,
            'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
        }
        annos.append(anno)

    data = {
        'filename': filename,
        'height': height,
        'width': width,
        'annos': annos
    }
    return data

def resize(image_path, anno_data, size):
    # size: (new_width, new_height)
    # size 크기로 맞춰 이미지 resize 및 bbox 정보 수정
    width = anno_data['width']
    height = anno_data['height']

    image = cv2.imread(image_path)
    image = cv2.resize(image, (size[0], size[1]))

    width_ratio = size[0] / width
    height_ratio = size[1] / height

    new_data = copy.deepcopy(anno_data)
    for anno in new_data['annos']:
        xmin, ymin = anno['bbox'][0], anno['bbox'][1]
        xmax, ymax = anno['bbox'][2], anno['bbox'][3]

        xmin, xmax = xmin * width_ratio, xmax * width_ratio
        ymin, ymax = ymin * height_ratio, ymax * height_ratio

        anno['bbox'] = [float(xmin), float(ymin), float(xmax), float(ymax)]

    return image, new_data

def json_to_yolo(anno_data):
    x_min = anno_data["annos"][0]["bbox"][0]
    y_min = anno_data["annos"][0]["bbox"][1]
    x_max = anno_data["annos"][0]["bbox"][2]
    y_max = anno_data["annos"][0]["bbox"][3]

    yolo_x = round(((x_min + x_max)/2) / int(anno_data["width"]), 6)
    yolo_y = round(((y_min + y_max)/2) / int(anno_data["height"]), 6)
    yolo_w = round((x_max - x_min) / int(anno_data["width"]), 6)
    yolo_h = round((y_max - y_min) / int(anno_data["height"]), 6)
    # print("yolo xywh" , yolo_x, yolo_y, yolo_w, yolo_h)

    # file name
    filename = anno_data['filename']

    # label
    label = anno_data["annos"][0]["label_num"]

    return filename, label, yolo_x, yolo_y, yolo_w, yolo_h

if __name__ == '__main__':
    main()
