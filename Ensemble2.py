from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
import os
from multiprocessing import Process
import numpy as np
from FAWDN.utils import util
import csv
import pickle

def start_FAWDN(lrpath, hrpath, modelpath,modelresulsts, modelnames, modelpsnr):
    # starts the fawdn test.py script and saves the results and which model produced them in lists
    srimages, srpsnr = FAWDN(lrpath, hrpath,modelpath)
    modelresulsts.append(srimages)
    modelpsnr.append(srpsnr)
    modelnames.append(modelpath.rsplit('/',1)[1]) #keeps track of when the models finished

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



def sortweigths(weigths, modelnames):
    # matches the results of models with their weigths, so the weights appear in the order the models finished in
    sortedweigths=[]
    for n1 in modelnames:
        #n1 = first_word(n1, 1)
        n1 = n1.rsplit('.',1)[0]
        for i in range(0, len(weigths)):
            n2 = weigths[i][1]
            #n2 = first_word(n2, 1)
            n2 = n2.rsplit('_',1)[0]

            if n1 == n2:
                sortedweigths.append(weigths[i][0])
                break
            elif i==len(weigths):
                print("did not match a csv file with model", n1,". Make sure csv file is named <model>_records.csv")
    return sortedweigths


def compute_ensemble(HRpath, weigths):
    # goes trough each resulting image from the models and computes a single ensemble image from them
    hrimages = [image for image in os.listdir(HRpath) if image.endswith('.png')]
    ensemblepsnr ,ensembleimages = [], []

    for hrimagenumber in range(0, len(hrimages)): # there is one result for each model for each HR image they tested on
        images = []
        # psnr = []
        for i in range(0, len(modelresults)):
                images.append(modelresults[i][hrimagenumber]) # gets SR result of model number i for testing image number hrimagenumber


        sortedweigths= sortweigths(weigths, modelnames)
        norm_psnr = []
        for value in sortedweigths:
            y = value  / (np.amax(sortedweigths))
            norm_psnr.append(y)
        imgs = np.array(images)

        mean_img = np.average(imgs, weights=norm_psnr, axis=0) # uses weights from validation testing
        #mean_img = imgs.mean(axis=0) # uses no weights

        mean_img = mean_img.astype('uint8')
        ensembleimages.append(mean_img)
        hr = cv2.imread(HRpath + "/" + hrimages[hrimagenumber]) # read the HR image to compare with emsemble image
        psnr, ssim = util.calc_metrics(mean_img, hr, crop_border=2) # computes psnr/ssim value of the ensemble
        ensemblepsnr.append((psnr,ssim))
    return ensembleimages,ensemblepsnr # returns lists of all the ensemble images and their psnr value

def perimageresults(modelresults,modelnames,modelpsnr):
    # prints the collected models results for each hr image in one line per image
    hrimages = [image for image in os.listdir(HRpath) if image.endswith('.png')]
    with open(os.getcwd()+ '/FAWDN/results/Modelpics/model.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for hrimagenumber in range(0, len(hrimages)):
            printtext = hrimages[hrimagenumber] +" PSNR/SSIM: "
            averagepsnr = []
            averagessim = []
            for i in range(0, len(modelresults)):
                #print(modelpsnr)
                #print(modelpsnr[i][hrimagenumber])
                psnr = modelpsnr[i][0][hrimagenumber] # [which model][psnr or ssim][image number]
                ssim = modelpsnr[i][1][hrimagenumber]  # ssim
                averagepsnr.append(psnr)
                averagessim.append(ssim)

                #psnr, ssim = modelpsnr[0][i][hrimagenumber]
                printtext +=str(modelnames[i])+" " + str(round(psnr, 2)) +" / "+ str(round(ssim, 2)) + " || "
                writer.writerow((modelnames[i], str(round(psnr, 2)), str(round(ssim, 2))))

            printtext += " Average " + str(round(np.mean(averagepsnr), 2))+" / "+str(round(np.mean(averagessim), 2))
            averagepsnr= str(np.mean(averagepsnr))
            averagessim = str(np.mean(averagessim))
            writer.writerow(("Average",averagepsnr,averagessim ))
            print(printtext)


def start_processes(json_files):
    # starts an amount of processes equal the amount of models provided
    processes= []
    for model in json_files:
       print("Running ", model)
       p = Process(target=start_FAWDN, args=(LRpath, HRpath, path_to_jsonfolder + model, modelresults, modelnames, modelpsnr))
       p.start()
       processes.append(p)
    return processes


if __name__ ==  '__main__':
    manager = mp.Manager()
    modelresults, modelnames, modelpsnr = manager.list(), manager.list(), manager.list()
    #sharedlistnames = manager.list()


    path_to_jsonfolder = os.getcwd()+"/FAWDN/options/final/" # current working directoy C:\Users\tobia\Documents\GitHub\VGIS8\
    LRpath = os.getcwd()+"/FAWDN/results/LR/MRI13/x2"
    HRpath = os.getcwd()+"/FAWDN/results/HR/MRI13/x2"

    json_files = [pos_json for pos_json in os.listdir(path_to_jsonfolder) if pos_json.endswith('.json')]
    processes = start_processes(json_files)

    for p in processes:
        p.join() #wait for all the processes to finish

    weigths = compute_weigths(path_to_jsonfolder)

    ensembleimages, ensemblepsnr = compute_ensemble(HRpath,weigths)

    with open(os.getcwd()+ '/FAWDN/results/Ensemble/ensemble.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for y in range (0,len(ensembleimages)):
            #cv2.imshow(str(y), ensembleimages[y])
            path=os.getcwd()+"/FAWDN/results/Ensemble/"+str(y+1)
            cv2.imwrite(path+".png", ensembleimages[y])
            psnr, ssim = ensemblepsnr[y]
            print("Ensemble image number" ,str(y+1), "PSNR/SSIM:",round(psnr,2),"/",round(ssim,2))
            psnr = str(round(psnr,2))
            ssim = str(round(ssim, 2))
            writer.writerow((y+1,psnr, ssim))





    perimageresults(modelresults, modelnames, modelpsnr)
    #print("ensemble psnr/ssim", psnr, ssim)

    for i in range(0,len(modelresults)):
        for j in range(0,len(modelresults[i])):
            path =os.getcwd()+"/FAWDN/results/Modelpics/"+str(modelnames[i])+"imagenum"+str(j+1)
            #cv2.imshow(str(modelnames[i])+"imagenum"+str(j+1), modelresults[i][j])
            cv2.imwrite(path+".png",modelresults[i][j])
    cv2.waitKey(0)
    #sum=sum/len(sharedlist)
    #print(sharedlist)

    #print()

