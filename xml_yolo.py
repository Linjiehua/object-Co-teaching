# 缺陷坐标xml转txt

import xml.etree.ElementTree as ET
import os
import shutil
import cv2
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
              'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
              'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']  # 输入缺陷名称，必须与xml标注名称一致

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

image_ids_train = open('/home/a123/lin/dataset/DIOR/test.txt').read().strip().split()  # 读取xml文件名索引
annotation_path = '/home/a123/lin/dataset/DIOR/Annotations'
img_path= "/home/a123/lin/dataset/DIOR/test/images"
label_save_path= img_path.replace('images','labels')
if os.path.exists(label_save_path):
    shutil.rmtree(label_save_path)
    os.makedirs(label_save_path)
else :
    os.makedirs(label_save_path)

def convert_annotation(image_id):
    in_file = open(os.path.join(annotation_path,'%s.xml' % (image_id)))  # 读取xml文件路径

    out_file = open(os.path.join(label_save_path,'%s.txt' % (image_id)), 'w')  # 需要保存的txt格式文件路径
    imgpath = os.path.join(img_path,image_id + ".jpg")
    img = cv2.imread(imgpath)
    imgInfo = img.shape
    h = imgInfo[0]
    w = imgInfo[1]
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:  # 检索xml中的缺陷名称
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    print(image_id)
for image_id in image_ids_train:
    convert_annotation(image_id)




