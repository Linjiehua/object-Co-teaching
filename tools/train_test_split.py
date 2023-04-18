from sklearn.model_selection import train_test_split
from glob import glob
import os
import cv2 as cv
import shutil

data_dir='/home/a309/lin/dataset/fair1m/'

img_dir=os.path.join(data_dir,'images')

label_dir=os.path.join(data_dir,'labels')


images_list=sorted(glob(img_dir+'/*.tif'))

trainset_dir='/home/a309/lin/dataset/fair1m/train1/'
valset_dir='/home/a309/lin/dataset/fair1m/test1/'
print(images_list)
imglist_train, imglist_test = train_test_split(images_list, test_size=0.5, random_state=2)

for _id,img in enumerate(imglist_train):
    print(img)
    img_id=img.split('/')[-1]
    label_name=img_id.replace('tif','txt')
    label=os.path.join(label_dir,label_name)
    print(label)
    trainimg_path=img.replace(data_dir,trainset_dir)
    trainlabel_path=label.replace(data_dir,trainset_dir)
    dir_im=os.path.dirname(trainimg_path)
    dir_label=os.path.dirname(trainlabel_path)
    if not os.path.exists(dir_im):
        os.makedirs(dir_im)
    if not os.path.exists(dir_label):
        os.makedirs(dir_label)
    if os.path.exists(label):
        shutil.copyfile(img, trainimg_path)
        shutil.copyfile(label,trainlabel_path)
for _id, img in enumerate(imglist_test):
    print(img)
    img_id = img.split('/')[-1]
    label_name = img_id.replace('tif', 'txt')
    label = os.path.join(label_dir, label_name)
    print(label)
    valimg_path = img.replace(data_dir, valset_dir)
    vallabel_path = label.replace(data_dir, valset_dir)
    dir_im = os.path.dirname(valimg_path)
    dir_label = os.path.dirname(vallabel_path)
    if not os.path.exists(dir_im):
        os.makedirs(dir_im)
    if not os.path.exists(dir_label):
        os.makedirs(dir_label)
    if os.path.exists(label):
         shutil.copyfile(img, valimg_path)
         shutil.copyfile(label, vallabel_path)
print(len(imglist_train),"###",len(imglist_test))
# for id,iteam in enumerate(iteams_list):
#     iteam_images=sorted(glob(iteam+'/*'))
#     for iteam_images_id,image_name in enumerate(iteam_images):
#         images_list.append(image_name)
# print(len(images_list))
# namelist_train, namelist_test = train_test_split(images_list, test_size=0.25, random_state=2)
# print('namelist_train',namelist_train)
# print('namelist_test',namelist_test)
# train_dir='./combine_data/train_set/'
# test_dir ='./combine_data/test_set/'
# for id,image_name in enumerate(namelist_train):
#     print(id)
#     save_name=image_name.replace(data_dir,train_dir)
#     iteam_name = save_name.split('/')[-2]
#     iteam_dir=train_dir+iteam_name+'/'
#     if not os.path.exists(iteam_dir):
#         print(iteam_dir)
#         os.makedirs(iteam_dir)
#     img=cv.imread(image_name)
#     cv.imwrite(save_name,img)
# print('test~~~~~~~~~~')
# for id,image_name in enumerate(namelist_test):
#     save_name=image_name.replace(data_dir,test_dir)
#     iteam_name = save_name.split('/')[-2]
#     iteam_dir=test_dir+iteam_name+'/'
#     if not os.path.exists(iteam_dir):
#         print(iteam_dir)
#         os.makedirs(iteam_dir)
#     img=cv.imread(image_name)
#     cv.imwrite(save_name,img)