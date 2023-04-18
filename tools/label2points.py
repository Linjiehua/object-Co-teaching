from glob import glob
import os
import numpy as np
import cv2 as cv
label_dir='/home/a123/lin/jskm1/val/labels/'
image_dir='/home/a123/lin/jskm1/val/images/'

images=sorted(glob(image_dir+'/*.png'))
label_list=sorted(glob(label_dir+'/*.txt'))

coco_label_dir='/home/a123/lin/jskm1/val/labels_p/'

if not os.path.exists(coco_label_dir):
    os.makedirs(coco_label_dir)
#print(label_list)
for _id,label in enumerate(label_list):

    image=images[_id]
    image_name = image.split('/')[-1].split('.')[0]
    print(image_name)
    label_name = label.split('/')[-1].split('.')[0]
    if not label_name == image_name:
        print('image name is not equal with label name ')
        continue
    im = cv.imread(image)
    im_height,im_width,chanel=64,1024,1
    with open(label,'r') as f:
        coco_label=label.replace(label_dir,coco_label_dir)
        with open(coco_label,'w') as f2:

            for _id_,line in enumerate(f.readlines()):
                line=line.split(' ')
                #print(line[0])
                p1_x=int(line[0])
                p1_y=int(line[1])
                p2_x=int(line[0])+int(line[2])
                p2_y=int(line[1])
                p3_x=int(line[0])
                p3_y=int(line[1])+int(line[3])
                p4_x=int(line[0])+int(line[2])
                p4_y=int(line[1])+int(line[3])
                ctr_x=(int(line[0])+int(line[2]))/(2*im_width)
                ctr_y=(int(line[1])+int(line[3]))/(2*im_height)
                height=int(line[3])/im_height
                width=int(line[2])/im_width
                f2.writelines(str(0)+' '+str(p1_x)+' '+str(p1_y)+' '+str(p2_x)+' '+str(p2_y)+' '+
                                         str(p3_x)+' '+str(p3_y)+' '+str(p4_x)+' '+str(p4_y)+'\n'  )
        #         print(top_x+width,top_y+height)
        #         im=cv.rectangle(im,(top_x,top_y),(top_x+width,top_y+height),color=(255,0,0))
        #         cv.circle(im, (ctr_x,ctr_y), radius=2,color=(0,0,255),thickness=5)
        # cv.imshow('result ',im)
        # cv.waitKey()
