import os
import numpy as np 
import math
import time
from matplotlib import pyplot as plt 
import cv2

def resize(images, size = (100,100)):
	images_norm = []
	for image in images:
		is_color = len(image.shape) == 3
		if is_color:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# using different OpenCV method if enlarging or shrinking
		if image.shape < size:
			image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
		else:
			image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
		images_norm.append(image_norm)

	return images_norm

def cut_faces(frame, face_coord):
	faces = []

	for (x, y, w, h) in face_coord:
		w_rm = int(0.3 * w / 2)
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

#start of the main function
print("Welcome")
folder = "People/" + raw_input('Enter the Persons Name: ').lower()

#capture the video and create windows
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live  Feed", 800, 400)

#Saving the video recording
fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('Videos/videos.avi', fourcc, 20.0, (640, 480))

#initialize the Haar Cascade
detector = cv2.CascadeClassifier("frontal_face.xml")

#check and save the images
if not os.path.exists(folder):
	print("Will start capturing.....")
	os.mkdir(folder)
	timer = 0
	counter = 0
	while counter < 30:
		#reading the frame and saving the video
		ret, frame = webcam.read()
		video.write(frame)

		#obtaining the face coordinates using the function and start the loop to save images
		face_cord = drawRect(frame, detector)
		if len(face_cord) and timer % 400 == 50:
			#crop the images respectively acc to the face coords
			cut_img = cut_faces(frame, face_cord)

			#convert the frame from BGR to GRAY
			gray = cv2.cvtColor(cut_img[0], cv2.COLOR_BGR2GRAY)

			#Normalizing the frame
			norm_face = cv2.equalizeHist(gray)

			print("Image " + str(counter) + " stored")
			counter = counter + 1

			#Resize the image to 100x100 matrix
			resized_face = cv2.resize(norm_face, (100, 100), interpolation = cv2.INTER_AREA)
			cv2.imshow("Face", resized_face)
			cv2.imwrite(folder + '/' + str(counter) + '.jpg', resized_face)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (0,0,0), 1, cv2.CV_AA)
		cv2.imshow("Live Feed", frame)
		cv2.waitKey(50)
		timer += 50

	cv2.destroyAllWindows()
	webcam.release()
	video.release()
else:
	print("This name exists")