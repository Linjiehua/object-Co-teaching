from glob import glob

image_dir='/home/a309/lin/dataset/SIMD/val/images'

images=sorted(glob(image_dir+'/*.jpg'))

print(images)
with open('/home/a309/lin/dataset/SIMD/val.txt','w') as f:
    for _id,image in enumerate(images):
        im_name=image.split('/')[-1].split('.')[0]
        path_img=im_name
        print(path_img)
        f.writelines(image+'\n')