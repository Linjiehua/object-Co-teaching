import torch
import numpy as np
import os
from glob import glob
import cv2

class_name =['car', 'truck', 'van', 'long-vehicle', 'bus', 'airliner', 'propeller-aircraft',
              'trainer-aircraft', 'chartered-aircraft', 'fighter-aircraft', 'others', 'stair-truck',
        'pushback-truck','helicopter', 'boat']
# class_name =['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
#               'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
#               'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
import cv2
from utils.plots import plot_one_box
from utils.general import xywh2xyxy ,scale_coords
from glob import glob
import os
import numpy as np
# gt_path='/home/a123/lin/dataset/DOTA/val_split/labels/'
# imgs_path = '/home/a123/lin/dataset/DOTA/val_split/images/'
gt_path='/home/a309/lin/yolo5_dior-5.3/runs_SIMD/val/expyolol/labels/'
imgs_path = '/home/a309/lin/dataset/SIMD/val/images/'
save_dir = '/home/a309/lin/yolo5_dior-5.3/runs_SIMD/val/expyolol/results/'
os.makedirs(save_dir,exist_ok=True)
img_list = sorted(glob(imgs_path+'*.jpg'))
gt_list =sorted(glob(gt_path+'*.txt'))
for _id,img in enumerate(img_list):
    # if _id>60:
    #     break
    img_name =img.split('/')[-1]
    print(img_name)
    im=cv2.imread(img)
    im_height,im_width,chanel=im.shape
    gt=img.replace(imgs_path,gt_path).replace('jpg','txt')
    if os.path.exists(gt):
        with open(gt) as f1:
            gt_boxs=f1.readlines()
            if gt_boxs==[]:
                print(img,' no object')

            else:
                for _id,box in enumerate(gt_boxs):
                    box = [float(i) for i in box.strip().split(' ') ]
                    cls = int(box[0])
                    box=np.array([box[1:]])
                   # print('xywh:',box)
                    xyxy=xywh2xyxy(box)
                    color=colors(cls)
                    #print('xyxy:',xyxy)
                    new_xyxy =[int(xyxy[0][0]*im_width),int(xyxy[0][1]*im_height),int(xyxy[0][2]*im_width),int(xyxy[0][3]*im_height)]
                    print('gt box:', new_xyxy)
                    plot_one_box(new_xyxy, im, color=color, label=class_name[cls],line_thickness=2)

    save_name=os.path.join(save_dir,img_name)
    cv2.imwrite(save_name,im)