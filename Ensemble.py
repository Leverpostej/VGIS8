from Bicubic.bicubic import main2 as bicubic
from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
from pathlib import Path

def start_bicubib(inputimage):

    r =  bicubic(inputimage)
    return r

def start_blindSR(inputimage):
    return bicubic(inputimage)

def start_DualSE(inputimage):
    return bicubic(inputimage)

def start_FAWDN(lrpath, hrpath): # fawdn takes a directory of lr images and a directory of hr images
    print("yo")
    r= FAWDN(lrpath, hrpath)
    return r


def collect_result(result):
    global resultimages
    for r in result:
        resultimages.append(r)

if __name__ ==  '__main__':
    pool = mp.Pool(mp.cpu_count())

    inputimageLR = cv2.imread("Datasets/MRI13/LR/1_ankle_LRBI_x2.png")
    inputimageHR = cv2.imread("Datasets/MRI13/HR/1_ankle_HR_x2.png")
    resultimages = []
    resultimages.append(start_FAWDN (str(Path().absolute())+"/Datasets/MRI13/LR/", str(Path().absolute())+"/Datasets/MRI13/HR"))

    #for i in range(2):
       #pool.apply_async(start_bicubib, args=(inputimageLR,), callback=collect_result)
       #pool.apply_async(start_FAWDN, args=(str(Path().absolute())+"/Datasets/MRI13/LR/",str(Path().absolute())+"/Datasets/MRI13/HR", ), callback=collect_result)
    #pool.close()
    #pool.join()
    #for i in range(0, len(resultimages)):
    #    image = resultimages[i]
    #    cv2.imshow("ost" + str(i), image)
    #cv2.waitKey(0)



    print(len(resultimages))

