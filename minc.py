import netCDF4 as nc
import cv2
import numpy as np
fn = 'LR.mnc'
ds = nc.Dataset(fn)
print(ds)



for i in range(0,100):
    print(ds['image'][i])
    cv2.imshow("image"+str(i), ds['image'][i])
    #cv2.imwrite("data/brainwebLR/image"+str(i), ds['image'][i])
cv2.waitKey(0)


#cv2.imwrite('color_img.jpg', img)
#cv2.imshow("image", img)

#prcp = ds['xspace'][0, 4000:4005, 4000:4005]