from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
import os
from multiprocessing import Process, Queue
import numpy as np
from PIL import Image
from scipy import misc
from FAWDN.utils import util
from sklearn.preprocessing import MinMaxScaler

def start_FAWDN(lrpath, hrpath, modelpath,sharedlist, sharedlistnames):
    sharedlist.append(FAWDN(lrpath, hrpath,modelpath))
    sharedlistnames.append(modelpath.rsplit('/',1)[1])

if __name__ ==  '__main__':
    manager = mp.Manager()
    sharedlist = manager.list()
    sharedlistnames = manager.list()
    #optionspath = ["normalpothoptions.json","C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/chromatic_FAWDN.json", "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/poisson_FAWDN.json"]

    path_to_jsonfolder = 'C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/'
    LRpath = "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/results/LR/MRI13/x2"
    HRpath = "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/results/HR/MRI13/x2"

    json_files = [pos_json for pos_json in os.listdir(path_to_jsonfolder) if pos_json.endswith('.json')]
    processes = []

    for model in json_files:
       print("Running ", model)
       p = Process(target=start_FAWDN, args=(LRpath, HRpath, path_to_jsonfolder+model, sharedlist, sharedlistnames))
       p.start()
       processes.append(p)

    for p in processes:
        p.join()

    hrimages =  [image for image in os.listdir(HRpath) if image.endswith('.png')]

    ensembleimages = []
    ensemblepsnr= []
    for hrimagenumber in range(0,len(hrimages)):
        images =[]
        #psnr = []
        for i in range(0, len(sharedlist)):
            if sharedlistnames[i] != "zbest_FAWDN+_x2.json":
                images.append(sharedlist[i][hrimagenumber])
                #psnr2, ssim = util.calc_metrics(sharedlist[i][0], hrimage, crop_border=2)
                #psnr.append(psnr2)

        #norm_psnr = []
        #for value in psnr:
        #    y = (value - np.amin(psnr)) / (np.amax(psnr) - np.amin(psnr))
        #    norm_psnr.append(y)

        #print(psnr)
        #print(norm_psnr)

        imgs = np.array(images)
        #mean_img = np.average(imgs, weights=norm_psnr, axis=0)
        mean_img = imgs.mean(axis=0)
        mean_img = mean_img.astype('uint8')
        ensembleimages.append(mean_img)
        hr = cv2.imread(HRpath+"/"+hrimages[hrimagenumber])
        psnr, ssim = util.calc_metrics(mean_img, hr, crop_border=2)
        ensemblepsnr.append(psnr)


    for y in range (0,len(ensembleimages)):
        cv2.imshow(str(y), ensembleimages[y])
        print("ensemble psnr/ssim", ensemblepsnr[y])

    #print("ensemble psnr/ssim", psnr, ssim)

    #for i in range(0,len(sharedlist)):
     #   cv2.imshow(sharedlistnames[i], sharedlist[i][0])

    cv2.waitKey(0)
    #sum=sum/len(sharedlist)
    #print(sharedlist)

    #print()

