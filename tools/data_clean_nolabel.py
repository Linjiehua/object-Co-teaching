from glob import glob
import os
import shutil
import numpy as np
import cv2 as cv
label_dir='/home/a309/lin/dataset/LEVIR/imageWithLabel/'
image_dir='/home/a309/lin/dataset/LEVIR/imageWithLabel/'
images=sorted(glob(image_dir+'/*.jpg'))
label_list=sorted(glob(label_dir+'/*.txt'))
#print(label_list)
for _id,label in enumerate(label_list):

    label_name = label.split('/')[-1].split('.')[0]
    print(label_name)
    img = os.path.join(image_dir,label_name+'.jpg')
    if os.path.exists(img):
        print(img)
    flag = False
    with open(label,'r') as f:
        #l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        l=f.read().strip().splitlines()
        if not len(l):
            flag=True
            if os.path.exists(img):
                os.remove(img)
                print('remove',img)
        else:
            flag=False
                #print(line)
    if flag:
        if os.path.exists(label):
            os.remove(label)
            print('remove', label)