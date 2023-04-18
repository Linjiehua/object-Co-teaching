from glob import glob
import os
import shutil
import numpy as np
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
              'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
              'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']  # 输入缺陷名称，必须与xml标注名称一致

img_path= "/home/a309/lin/dataset/DIOR_40nc/train/images"
label_path= img_path.replace('images','labels_o')
label_save_path= img_path.replace('images','labels_0.4n')
if os.path.exists(label_save_path):
    shutil.rmtree(label_save_path)
    os.makedirs(label_save_path)
else :
    os.makedirs(label_save_path)
noise_rate = 0.4

label_list = glob(label_path+'/*.txt')

for _id, label in enumerate(label_list):
    with open(label,'r') as f:
        label_noise = label.replace('labels_o','labels_0.4n')
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
                other_class_id = np.delete(class_id,cls_id)
                cls_id_n = np.random.choice(other_class_id)
                out_file.write(str(cls_id_n) + ' ' + str(ctr_x) + ' ' + str(ctr_y) + ' ' + str(w) + ' ' + str(h) + '\n')
                #print(str(cls_id_n) + ' ' + str(ctr_x) + ' ' + str(ctr_y) + ' ' + str(w) + ' ' + str(h) + '\n')
            else:
                out_file.write(line_o)
                #print(line_o)


