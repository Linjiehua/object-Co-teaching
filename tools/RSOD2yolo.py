from glob import glob
import os
import numpy as np
import cv2 as cv
class_names=("aircraft","oiltank","overpass","playground")
label_dir='/home/a309/lin/dataset/RSOD/train/labels_o/'
image_dir='/home/a309/lin/dataset/RSOD/train//images/'

images=sorted(glob(image_dir+'/*.jpg'))
label_list=sorted(glob(label_dir+'/*.txt'))

coco_label_dir='/home/a309/lin/dataset/RSOD/train/labels/'

if not os.path.exists(coco_label_dir):
    print('###')
    os.makedirs(coco_label_dir)
#print(label_list)
for _id,label in enumerate(label_list):



    label_name = label.split('/')[-1].split('.')[0]
    print(label_name)
    image = os.path.join(image_dir,label_name+'.jpg')
    # if not label_name == image_name:
    #     print('image name is not equal with label name ')
    #     continue
    im = cv.imread(image)
    im_height,im_width,chanel=im.shape
    with open(label,'r') as f:
        coco_label=label.replace(label_dir,coco_label_dir)
        with open(coco_label,'w') as f2:

            for _id_,line in enumerate(f.readlines()):
                if line=='':
                    continue
                line=line.split('\t')


                print(line)
                class_name=line[1]
                class_id=class_names.index(class_name)
               # print(class_id)
                ctr_x=((int(line[2])+int(line[4]))/2)/im_width
                ctr_y=((int(line[3])+int(line[5]))/2)/im_height
                height=(int(line[5])-int(line[3]))/im_height
                width=(int(line[4])-int(line[2]))/im_width
                f2.writelines(str(class_id)+' '+str(ctr_x)+' '+str(ctr_y)+' '+str(width)+' '+str(height)+'\n')
        #         print(top_x+width,top_y+height)
        #         im=cv.rectangle(im,(top_x,top_y),(top_x+width,top_y+height),color=(255,0,0))
        #         cv.circle(im, (ctr_x,ctr_y), radius=2,color=(0,0,255),thickness=5)
        # cv.imshow('result ',im)
        # cv.waitKey()
