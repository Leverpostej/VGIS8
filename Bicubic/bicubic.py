import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    names =[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #images.append(img)
            img_resized = cv2.resize(img, (30,30), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(folder,filename), img_resized)
            #names.append(os.path.join(filename))
        else:
            print("image is none")
    return images, names

def main2(inputimage):

    image =inputimage

    return cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)



def main():
    vgisfolder= os.path.dirname(__file__)
    images, names = load_images_from_folder("C:/Users/tobia/Desktop/images/Blurred")
    i=0
    #np.random.shuffle(images)

    dim = (30, 30)
    out=[]
    for image in images:
        #img_resized= cv2.resize(image,dim,interpolation=cv2.INTER_CUBIC)
        #blurred = cv2.GaussianBlur(image,(3,3),0)
        blurred2 = cv2.GaussianBlur(image, (7, 7), 0)

        sandp= noisy("s&p", image)
        poisson = noisy("poisson", image)
        speckle = noisy("speckle", image)
        #cv2.imshow("blur", blurred)
        #cv2.imshow("before", blurred2)
        #cv2.waitKey()
        #out.append(img_resized)
        #out.append(img_resized)
        #cv2.imshow("before", image)

        #cv2.waitKey(0)
        print(names[i])
        #cv2.imwrite("C:/Users/tobia/Desktop/images/Salty/"+names[i], sandp)
        #cv2.imwrite("C:/Users/tobia/Desktop/images/Poisson/" + names[i], poisson)
        #cv2.imwrite("C:/Users/tobia/Desktop/images/Speckle/" + names[i], speckle)
        i+=1
    #print("hej")

    #train = out[:round(len(images)-(len(images)/3))]
    #val = out[round(len(images)-(len(images)/3)):]
    #print(len(train))
    #print(len(val))
    #i = 1
    #for image in train:
    #    cv2.imwrite("C:/Users/tobia/Desktop/images/HR/" + str(+i) + ".png", image)
    #    i+=1
    #i = 1
    #for image in val:
    #    cv2.imwrite("C:/Users/tobia/Desktop/images/Val/HR/" + str(+i) + ".png", image)
    #    i += 1
   # return out

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.04
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


#main()