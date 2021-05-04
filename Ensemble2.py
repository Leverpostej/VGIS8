from FAWDN.test import main as FAWDN
import cv2
import multiprocessing as mp
import os
from multiprocessing import Process, Queue


def start_FAWDN(modelpath,sharedlist):
    sharedlist.append(FAWDN(modelpath))

if __name__ ==  '__main__':
    manager = mp.Manager()
    sharedlist = manager.list()

    #optionspath = ["normalpothoptions.json","C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/chromatic_FAWDN.json", "C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/poisson_FAWDN.json"]

    path_to_jsonfolder = 'C:/Users/tobia/Documents/GitHub/VGIS8/FAWDN/options/final/'
    json_files = [pos_json for pos_json in os.listdir(path_to_jsonfolder) if pos_json.endswith('.json')]
    processes = []

    for model in json_files:
       print("Running ", model)
       p = Process(target=start_FAWDN, args=(path_to_jsonfolder+model,sharedlist, ))
       p.start()
       processes.append(p)

    for p in processes:
        p.join()



    print(sharedlist)

    print(len(sharedlist))

