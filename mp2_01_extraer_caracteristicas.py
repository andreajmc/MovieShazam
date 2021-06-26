import os
import sys
import numpy as np
import cv2 
import pickle
from vectors import Point, Vector
from cv2 import FastFeatureDetector
from matplotlib import pyplot as plt

images = []

#path = r"C:\Users\Daniel\Documents\Sistemas Inteligentes\Mini Proyecto 2\training_image_dataset\training_image_dataset"
#path1 = r"C:\Users\Daniel\Documents\Sistemas Inteligentes\Mini Proyecto 2\X\Img"


path1 = sys.argv[1]
path2 = sys.argv[2]
Images = {}
for file in os.listdir(path1):
    print ('Extrayendo caracteristicas del archivo: ',file)
    img1 = cv2.imread(path1+"\\"+file)
    width = 128
    height = 256
    dim = (width, height)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img_og =img1
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img1 = cv2.equalizeHist(img1)
    img1 = cv2.medianBlur(img1,7)
    #Eliminar ruido
    th2 = cv2.adaptiveThreshold(img1,250,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,11,2)

    #Otro metodo para sacar los thresholds y eliminar ruido            
    #th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY_INV,11,2)

    #Imagen original       
    im =cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    #Median Adaptive Threshold
    mean = cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)
    #Restar Median de la Original
    subs =cv2.subtract(im,mean)
    
    #final = cv2.bitwise_and(im, mean)
    #Diferencia absoluta de la Median y la Original 
    #dist = cv2.absdiff(im, mean)

    #Cuadros para buscar el dinero
    contours,hierarchy = cv2.findContours(th2,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    #Extrae el cuadro mas grande
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img_og,
                        (x,y),(x+w,y+h),
                        (0,255,0),
                        2)

    #Recortar imagen apartir del cuadro
    x,y,w,h = cv2.boundingRect(c)
    ROI = img_og[y:y+h, x:x+w]

    #Extrar caracteristicas
    hog = cv2.HOGDescriptor()
    im1 = ROI
    width = 128
    height = 256
    dim = (width, height)
    im1 = cv2.resize(ROI, dim, interpolation = cv2.INTER_AREA)
    h = hog.compute(im1)
    arrayFet = []
    #cv2.imwrite(r"C:\Users\Daniel\Documents\Sistemas Inteligentes\Mini Proyecto 2\X"+'\\'+str(file),im1)
    for x in h:
        arrayFet.append(x[0])   
    X = np.array(arrayFet)
    Images[file] = X




with open(path2, 'wb') as fp:
        pickle.dump(Images, fp)
np.save('features.npy', Images)
