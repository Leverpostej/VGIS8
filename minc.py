import netCDF4 as nc
import cv2
import numpy as np
import os
from PIL import Image

vgisfolder= os.path.dirname(__file__)
fn = '/Datasets/ADNI/'
ds = nc.Dataset(vgisfolder+fn+"ADNI1Baseline_3T.mnc")
print(len(ds['image']))



for i in range(0,len(ds['image'])):
    #print(ds['image'][i])
    #cv2.imshow("image"+str(i), ds['image'][i])
    img = Image.fromarray(ds['image'][i])
    img = img.convert("L")
    img.save(vgisfolder+fn+"pngs/ADNI_"+str(i)+".png", "PNG")
#    print(vgisfolder+fn+"pngs/ADNI_"+str(i)+".png")
    #cv2.imwrite(vgisfolder+fn+"/pngs/LR/lr_"+str(i)+".jpg", ds['image'][i])
cv2.waitKey(0)


#cv2.imwrite('color_img.jpg', img)
#cv2.imshow("image", img)

#prcp = ds['xspace'][0, 4000:4005, 4000:4005]