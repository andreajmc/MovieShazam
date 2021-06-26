import os
import sys
import numpy as np
import cv2 


def preprocess():
    dim = (640,360)
    image = cv2.imread(sys.argv[1])
    #copy = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    copy = image.copy()
    gray =cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray,7)
    #edges = cv2.Canny(gray,0,50)
    

    threshold = cv2.adaptiveThreshold(gray,250,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    
    contours,hierarchy = cv2.findContours(threshold.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    
    arrayAreas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        arrayAreas.append(area)
    
    sortAreas = sorted(contours, key=cv2.contourArea,reverse=True)
    largestArea = sortAreas[0]
    x,y,w,h = cv2.boundingRect(largestArea)
    

    
    #x,y,w,h = cv2.boundingRect(secondlargestcontour)
    #cv2.rectangle(copy,
    #                    (x,y),(x+w,y+h),
    #                    (0,255,0),
    #                    2)
    #x,y,w,h = cv2.boundingRect(screen)
    img_result = copy[y:y+h, x:x+w]
    print(copy)
    img_result= cv2.resize(img_result, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("copy",copy)
    cv2.imshow("thres",threshold)
    cv2.imshow("result",img_result)
    
    cv2.waitKey(0)
    
    cv2.imwrite("img_result.png", img_result)
preprocess()