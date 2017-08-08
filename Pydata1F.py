import os
import numpy as np 
import math
from matplotlib import pyplot as plt 
import cv2

def cut_faces(frame, face_coord):
	faces = []

	for (x, y, w, h) in face_coord:
		w_rm = int(0.2 * w / 2)
		faces.append(frame[y: y + h, x + w_rm: x + w - w_rm])

	return faces


def drawRect(frame, detector):
	biggest = True
	scale_factor = 1.2
	min_neighbors = 5
	min_size = (30, 30)

	flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
    		cv2.CASCADE_DO_ROUGH_SEARCH if biggest else \
    		cv2.CASCADE_SCALE_IMAGE 

	face_coord = detector.detectMultiScale(
            frame,
            scaleFactor = scale_factor,
            minNeighbors = min_neighbors,
            minSize = min_size,
            flags = flags
        )

	#print face_coord
	for (x, y, w, h) in face_coord:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 0), 5)

	return face_coord

#start of the video
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cut images", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live  Feed", 800, 400)

#Saving the video recording
fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('../../Videos/videos.avi', fourcc, 20.0, (640, 480))

#initialize cascade classifier
detector = cv2.CascadeClassifier("frontal_face.xml")

#main loop
try:
	while 1:
	    ret, frame = webcam.read()
	    video.write(frame)

	    face_cord = drawRect(frame, detector)

	    if len(face_cord):
			cut_img = cut_faces(frame, face_cord)
			gray = cv2.cvtColor(cut_img[0], cv2.COLOR_BGR2GRAY)
			norm_face = cv2.equalizeHist(gray)
			vis = np.concatenate((gray, norm_face), axis=1)
			cv2.imshow("Cut images", vis)

	    font = cv2.FONT_HERSHEY_SIMPLEX
	    cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (0,0,0), 1, cv2.CV_AA)
	    cv2.imshow("Live Feed", frame)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
except KeyboardInterrupt:
	print "Live Video Interrupted"

cv2.destroyAllWindows()
webcam.release()
video.release()