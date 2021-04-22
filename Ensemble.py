from Bicubic.bicubic import main2 as bicubic
import cv2
import multiprocessing as mp


def start_bicubib(inputimage):
    print("hejsa")
    r =  bicubic(inputimage)
    return r

def start_blindSR(inputimage):
    return bicubic(inputimage)

def start_DualSE(inputimage):
    return bicubic(inputimage)

def start_FAWDN(inputimage):
    return bicubic(inputimage)


def collect_result(result):
    global resultimages
    resultimages.append(result)

if __name__ ==  '__main__':
    pool = mp.Pool(mp.cpu_count())

    inputimage = cv2.imread("Datasets/MRI13/LR/1_ankle_LRBI_x2.png")

    resultimages = []

    for i in range(10):
        pool.apply_async(start_bicubib, args=(inputimage,), callback=collect_result)

    pool.close()
    pool.join()




    print(len(resultimages))

