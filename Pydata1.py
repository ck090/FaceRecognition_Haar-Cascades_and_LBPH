import os
import numpy as np 
import math
from matplotlib import pyplot as plt 
import cv2

webcam = cv2.VideoCapture(0)

while(1):

    ret, frame = webcam.read()
    print ret

    cv2.startWindowThread()

    #cv2.namedWindow("First", cv2.WINDOW_NORMAL)
    cv2.imshow("TUTS", frame)

cv2.waitKey()
cv2.destroyAllWindows()
webcam.release()