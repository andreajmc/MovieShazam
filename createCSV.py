import cv2
from os import listdir
from os.path import isfile, join
import sys

dim = (152, 84)

def create_csv():
for f in listdir(sys.argv[1]):
    if isfile:
        path = sys.argv[1] + "/" + f
        image = cv2.imread(path)
        resized = cv2.resize(image,dim)
        cv2.imshow("resized", resized)
        cv2.imwrite(f,resized)