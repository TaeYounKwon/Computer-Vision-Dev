import os, glob
a,b = [], []

train_image_path = f"D:/datasets/train/images"
train_image = glob.glob(os.path.join(train_image_path, "*.jpg"))
for train_img in train_image :
    train_img = os.path.basename(train_img).split('\\')[-1].replace('.jpg', '')
    a.append(train_img)
a = sorted(a)

train_label_path = f"D:/datasets/train/labels"
train_label = glob.glob(os.path.join(train_label_path, "*.text"))
for train_lab in train_label :
    train_lab = os.path.basename(train_lab).split('\\')[-1].replace('.text', '')
    b.append(train_lab)
b = sorted(b)


c, d = [], []
train_image_path = f"D:/datasets/val/images"
train_image = glob.glob(os.path.join(train_image_path, "*.jpg"))
for train_img in train_image :
    train_img = os.path.basename(train_img).split('\\')[-1].replace('.jpg', '')
    c.append(train_img)
c = sorted(c)

train_label_path = f"D:/datasets/val/labels"
train_label = glob.glob(os.path.join(train_label_path, "*.text"))
for train_lab in train_label :
    train_lab = os.path.basename(train_lab).split('\\')[-1].replace('.text', '')
    d.append(train_lab)
d = sorted(d)

# print(len(set(d) - set(c)))
# print(list(set(c) - set(d)))


image_name_list_train = list(set(a) - set(b))
image_name_list_val = list(set(c) - set(d))

for item in image_name_list_train : # 기존 34463 | 변경 후 34334
    os.remove(f'D:/datasets/train/images/{item}.jpg')
# print(len(set(a) - set(b)))

for item in image_name_list_val : # 기존 4226 | 변경 후 4210
    os.remove(f'D:/datasets/val/images/{item}.jpg')
# print(len(set(c) - set(d)))