from PIL import Image

from Bicubic.bicubic import main2 as bicubic
from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
from pathlib import Path
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        #img = cv2.imread(os.path.join(folder,filename))
        img = cv2.imread(os.path.join(folder,filename))
        #if img is not None:
        images.append(img)
    return images

def prepareData():
    path = "C:/Users/tobia/Desktop/val"
    images = load_images_from_folder(path)
    imgset=np.array(images)
    #np.save("C:/Users/tobia/Desktop/val/npy/val.npy",imgset)

    #np.random.shuffle(images)
    #split =np.random.randint(len(images)/-(len(images)/3))
    #teststet = images[:split]
    #valiset = images[split:]

    # turns it into npy

def start_FAWDN(hrpath, lrpath, modelpath): # fawdn takes a directory of lr images and a directory of hr images
    print("yo")
    r= FAWDN(hrpath, lrpath, modelpath)
    return r


def collect_result(result):
    global resultimages
    for r in result:
        resultimages.append(r)

if __name__ ==  '__main__':
    #pool = mp.Pool(mp.cpu_count())
    #prepareData()
    #f = np.load("C:/Users/tobia/Desktop/downloads/npy/imgds.npy", allow_pickle=True)

    hrset = "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/results/HR/MRI13/x2"
    lrset = "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/results/LR/MRI13/x2"
    results = []
    #optionspath = ["normalpothoptions.json","C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/chromatic_FAWDN.json", "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/poisson_FAWDN.json"]
    path_to_jsonfolder = 'C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/'
    json_files = [pos_json for pos_json in os.listdir(path_to_jsonfolder) if pos_json.endswith('.json')]
    for model in json_files:
       print("Running ", model)
       results.append(start_FAWDN(hrset, lrset, path_to_jsonfolder+model))

    #inputimageLR = cv2.imread("Datasets/MRI13/LR/1_ankle_LRBI_x2.png")
    #inputimageHR = cv2.imread("Datasets/MRI13/HR/1_ankle_HR_x2.png")
    #resultimages = []
    #resultimages.append(start_FAWDN (str(Path().absolute())+"/Datasets/MRI13/LR/", str(Path().absolute())+"/Datasets/MRI13/HR"))

    #for i in range(2):
       #pool.apply_async(start_bicubib, args=(inputimageLR,), callback=collect_result)
       #pool.apply_async(start_FAWDN, args=(str(Path().absolute())+"/Datasets/MRI13/LR/",str(Path().absolute())+"/Datasets/MRI13/HR", ), callback=collect_result)
    #pool.close()
    #pool.join()
    #for i in range(0, len(resultimages)):
    #    image = resultimages[i]
    #    cv2.imshow("ost" + str(i), image)
    #cv2.waitKey(0)

    print(results)
    print(len(results))

