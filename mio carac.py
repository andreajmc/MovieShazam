import cv2
import numpy as np
import pickle
import os
import sys

def extract_features(image_path, vector_size=32):
    width = 320
    height = 240
    dim = (width, height)

    image = cv2.imread(image_path)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    
    contours,hierarchy = cv2.findContours(thresh,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(copy,
                        (x,y),(x+w,y+h),
                        (0,255,0),
                        2)

    x,y,w,h = cv2.boundingRect(c)
    ROI = copy[y:y+h, x:x+w]
    
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('copy', ROI)
    #cv2.waitKey()

    image = ROI
    
    sift = cv2.SIFT_create()
    
    kps = sift.detect(image)

    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = sift.compute(image, kps)
    #dsc = dsc.flatten()
    
    
    #cv2.imshow('copy', hsv)
    #cv2.waitKey()
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
    
    
    #features.append(hist)
    features = np.array(features)
    features = features.flatten()

    dsc = dsc.flatten()
    dsc = np.concatenate([dsc,np.zeros(1)])

    finalArray = np.concatenate([features,dsc])
    finalArray = np.array(finalArray)
    
    return finalArray

def main():
    images_path=sys.argv[1]
    pickled_db_path = sys.argv[2]
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        name = f.split('\\')[-1]
        print ('Extracting features from image %s' % name)
        result[name] = extract_features(f)
    
    with open(pickled_db_path, 'wb') as fp:
        
        pickle.dump(result, fp)
    
        


main()
