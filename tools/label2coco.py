from glob import glob
import os
import numpy as np
import cv2 as cv
label_dir='/home/a123/js_dataset/val/rbox_labels/'
image_dir='/home/a123/js_dataset/val/images/'

images=sorted(glob(image_dir+'/*.tif'))
label_list=sorted(glob(label_dir+'/*.txt'))

coco_label_dir='/home/a123/js_dataset/val/coco_label/'

if not os.path.exists(coco_label_dir):
    os.makedirs(coco_label_dir)
print(label_list)
for _id,label in enumerate(label_list):

    image=images[_id]
    image_name = image.split('/')[-1].split('.')[0]
    label_name = label.split('/')[-1].split('.')[0]
    if not label_name == image_name:
        print('image name is not equal with label name ')
        continue
    im = cv.imread(image)
    im_height,im_width,chanel=im.shape
    with open(label,'r') as f:
        coco_label=label.replace(label_dir,coco_label_dir)
        with open(coco_label,'w') as f2:

            for _id_,line in enumerate(f.readlines()):
                line=line.split(' ')
                box=[[int(line[1]),int(line[2])],[int(line[3]),int(line[4])],
                     [int(line[5]),int(line[6])],[int(line[7]),int(line[8])]]
                box = np.array(box)
                top_x=min(box[:,0])
                top_y=min(box[:,1])
                ctr_x=int((min(box[:,0])+max(box[:,0]))/2)/im_width
                ctr_y=int((min(box[:,1])+max(box[:,1]))/2)/im_height
                height=max(box[:,1]-min(box[:,1]))/im_height
                width=max(box[:,0]-min(box[:,0]))/im_width
                print(str(int(line[0])-1))
                f2.writelines(str(int(line[0])-1)+' '+str(ctr_x)+' '+str(ctr_y)+' '+str(width)+' '+str(height)+'\n')
        #         print(top_x+width,top_y+height)
        #         im=cv.rectangle(im,(top_x,top_y),(top_x+width,top_y+height),color=(255,0,0))
        #         cv.circle(im, (ctr_x,ctr_y), radius=2,color=(0,0,255),thickness=5)
        # cv.imshow('result ',im)
        # cv.waitKey()
