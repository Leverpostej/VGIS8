import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

vgisfolder= os.path.dirname(__file__)
images = load_images_from_folder(vgisfolder+ "/Datasets/MRI13/LR")
i=1
for image in images:
    img_resized= cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("Results/mri13_"+str(+i)+".png", img_resized)
    i+=1
