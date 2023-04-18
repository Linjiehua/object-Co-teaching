from glob import glob
import os
import shutil
import numpy as np
classes = ["aircraft","oiltank","overpass","playground"]  # 输入缺陷名称，必须与xml标注名称一致

img_path= "/home/a309/lin/dataset/RSOD_10nc/train/images"
label_path= img_path.replace('images','labels_o')
label_save_path= img_path.replace('images','labels')
if os.path.exists(label_save_path):
    shutil.rmtree(label_save_path)
    os.makedirs(label_save_path)
else :
    os.makedirs(label_save_path)
noise_rate = 0.1

label_list = glob(label_path+'/*.txt')
a=len(label_list)
j=0
i=0
for _id, label in enumerate(label_list):
    with open(label,'r') as f:
        label_noise = label.replace('labels_o','labels')
        out_file = open(label_noise, 'w')
        print(label)
        print(label_noise)
        for _id_, line_o in enumerate(f.readlines()):
            line = line_o.split(' ')
            cls_id = int(line[0])
            ctr_x = float(line[1])
            ctr_y = float(line[2])
            w = float(line[3])
            h = float(line[4])
            n = np.random.random()
            #print("origin :",line_o)
            if n<=noise_rate:
                class_id = list(range(len(classes)))
               # print(class_id)
                j=j+1
                i=i+1
                other_class_id = np.delete(class_id,cls_id)
                cls_id_n = np.random.choice(other_class_id)
                out_file.write(str(cls_id_n) + ' ' + str(ctr_x) + ' ' + str(ctr_y) + ' ' + str(w) + ' ' + str(h) + '\n')
                #print(str(cls_id_n) + ' ' + str(ctr_x) + ' ' + str(ctr_y) + ' ' + str(w) + ' ' + str(h) + '\n')
            else:
                j=j+1
                out_file.write(line_o)
                #print(line_o)

print(a)
print(i)
print(j)
print(i/j)