import cv2 as cv
from glob import glob
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
# label_dir='../../jskm1/train/bz/'
# image_dir='../../jskm1/train/images3/'
# save_dir='../../jskm1/train/detection4/'
label_dir='/home/a309/lin/dataset/DIOR_50nc/train/labels/'
image_dir='/home/a309/lin/dataset/DIOR_50nc/train/images/'
save_dir = '/home/a309/lin/dataset/DIOR_50nc/train/noise_img/'
# img='/home/a123/lin/km1/tif原始数据图片/train/images/1.png'
# im=Image.open(img)
# print(im.size)
# im=np.array(im)
# print(np.min(im),np.max(im))
images=sorted(glob(image_dir+'/*.jpg'))
print(images)
labels=sorted(glob(label_dir+'/*.txt'))
print(labels)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for _id,image in enumerate(images):
    #label=labels[_id]
    image_name=image.split('/')[-1].split('.')[0]
    label_name=image_name+'.txt'
    image_n=image_name+'.jpg'
    save_name=os.path.join(save_dir,image_n)
    #image_name=image.split('/')[-1].split('.')[0]
    label=os.path.join(label_dir,label_name)
    save_name =os.path.join(save_dir,image_n)
    print('img_name:',image,'label_name:',label_name)
    # if not label_name==image_name:
    #     print('image name is not equal with label name ')
    #     continue
    if not os.path.isfile(label):
        continue
    im=cv.imread(image)
    #im=cv.transpose(im)
    with open(label,'r') as f:
        for _id_,line in enumerate(f.readlines()):
            if line==' ':
                continue
            line=line.split(' ')
            #box=[int(line[0]),int(line[1]),int(line[2]),int(line[3])]
            box = [int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4]))]
            box=np.array(box)
            #print(box)
            #im = cv.rectangle(im, (box[1],box[0]), (box[1] + box[3],box[0] + box[2]), (255, 0, 255))
            im=cv.rectangle(im,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,255))
    cv.imwrite(save_name,im)
