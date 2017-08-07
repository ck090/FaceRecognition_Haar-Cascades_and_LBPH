import os
import numpy as np 
import math
from matplotlib import pyplot as plt 
import cv2

webcam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live  Feed", 300, 400)

while 1:
    ret, frame = webcam.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
webcam.release()