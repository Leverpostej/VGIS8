from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
import os
from multiprocessing import Process
import numpy as np
from FAWDN.utils import util
import csv


def start_FAWDN(lrpath, hrpath, modelpath,modelresulsts, modelnames):
    # starts the fawdn test.py script and saves the results and which model produced them in lists
    modelresulsts.append(FAWDN(lrpath, hrpath,modelpath))
    modelnames.append(modelpath.rsplit('/',1)[1])

def compute_weigths(path_to_jsonfolder):
    # finds the cvs files in the directory and means the psnr values for each file
    excelfiles = [excel for excel in os.listdir(path_to_jsonfolder) if excel.endswith('.csv')]
    meanpsnr=[]
    for file in excelfiles:
        filepsnr=[]
        with open(path_to_jsonfolder+"/"+file, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row is not None:
                    filepsnr.append(row["psnr"])
            filepsnr = np.array(filepsnr).astype(np.float)
            t = np.mean(filepsnr),file
            meanpsnr.append(t)

    return meanpsnr # contains a list of (meanpsnr, modelname) for each csv file

def compute_ensemble(HRpath):
    # goes trough each resulting image from the models and computes a single ensemble image from them
    hrimages = [image for image in os.listdir(HRpath) if image.endswith('.png')]
    ensemblepsnr ,ensembleimages = [], []

    for hrimagenumber in range(0, len(hrimages)): # there is one result for each model for each HR image they tested on
        images = []
        # psnr = []
        for i in range(0, len(modelresulsts)):
            if modelnames[i] != "zbest_FAWDN+_x2.json":  #exclude fawdn's own nest model from the emsenble
                images.append(modelresulsts[i][hrimagenumber]) # gets SR result of model number i for testing image number hrimagenumber
                # psnr2, ssim = util.calc_metrics(sharedlist[i][0], hrimage, crop_border=2)
                # psnr.append(psnr2)

        # norm_psnr = []
        # for value in psnr:
        #    y = (value - np.amin(psnr)) / (np.amax(psnr) - np.amin(psnr))
        #    norm_psnr.append(y)

        # print(psnr)
        # print(norm_psnr)

        imgs = np.array(images)
        # mean_img = np.average(imgs, weights=norm_psnr, axis=0)
        mean_img = imgs.mean(axis=0)
        mean_img = mean_img.astype('uint8')
        ensembleimages.append(mean_img)
        hr = cv2.imread(HRpath + "/" + hrimages[hrimagenumber]) # read the HR image to compare with emsemble image
        psnr, ssim = util.calc_metrics(mean_img, hr, crop_border=2) #computes psnr value of the ensemble
        ensemblepsnr.append(psnr)
    return ensembleimages,ensemblepsnr # returns lists of all the ensemble images and their psnr value


def start_processes(json_files):
    # starts an amount of processes equal the amount of models provided
    processes= []
    for model in json_files:
       print("Running ", model)
       p = Process(target=start_FAWDN, args=(LRpath, HRpath, path_to_jsonfolder + model, modelresulsts, modelnames))
       p.start()
       processes.append(p)
    return processes


if __name__ ==  '__main__':
    manager = mp.Manager()
    modelresulsts, modelnames = manager.list(), manager.list()
    #sharedlistnames = manager.list()


    path_to_jsonfolder = 'C:/Users/tobias/Documents/GitHub/VGIS8/final/'
    LRpath = "C:/Users/tobias/Documents/GitHub/VGIS8/FAWDN/results/LR/MRI13/x2"
    HRpath = "C:/Users/tobias/Documents/GitHub/VGIS8/FAWDN/results/HR/MRI13/x2"

    json_files = [pos_json for pos_json in os.listdir(path_to_jsonfolder) if pos_json.endswith('.json')]
    #processes = start_processes(json_files)

    #for p in processes:
    #    p.join()

    weigths = compute_weigths("D:/final")
    #ensembleimages, ensemblepsnr = compute_ensemble(HRpath)

    #for y in range (0,len(ensembleimages)):
    #    cv2.imshow(str(y), ensembleimages[y])
    #    print("ensemble psnr/ssim", ensemblepsnr[y])

    #print("ensemble psnr/ssim", psnr, ssim)

    #for i in range(0,len(sharedlist)):
     #   cv2.imshow(sharedlistnames[i], sharedlist[i][0])

    #cv2.waitKey(0)
    #sum=sum/len(sharedlist)
    #print(sharedlist)

    #print()

