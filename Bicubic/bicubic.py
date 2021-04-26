import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def main2(inputimage):

    image =inputimage

    return cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)



def main():
    vgisfolder= os.path.dirname(__file__)
    images = load_images_from_folder("C:/Users/Tobias/Documents/GitHub/VGIS8/Datasets/MRI13/LR")
    i=1
    out=[]
    for image in images:
        img_resized= cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        out.append(img_resized)
        #cv2.imwrite("Results/mri13_"+str(+i)+".png", img_resized)
        i+=1
    print("hej")
    return out