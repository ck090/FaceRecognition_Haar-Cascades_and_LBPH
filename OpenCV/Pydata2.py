import os
import numpy as np 
import math
import time
from matplotlib import pyplot as plt 
import cv2
import pyttsx
import time

def draw_face_ellipse(image, faces_coord):
#Draws an ellipse around the face found.
	for (x, y, w, h) in faces_coord:
		center = (x + w / 2, y + h / 2)
		axis_major = h / 2
		axis_minor = w / 2
		cv2.ellipse(image,
		        center=center,
		        axes=(axis_major, axis_minor),
		        angle=0,
		        startAngle=0,
		        endAngle=360,
		        color=(206, 0, 209),
		        thickness=2)
	return image

""" Function to collect the dataset pre-saved from the other python code
	This opens the folder saved and loops through each sub-folder in the
	main folder which is the name of the person.
	It returns the images, numpy array of labels, and a label-dictionary
"""
def collect_dataset():
	images = []
	labels = []
	labels_dic = {}
	people = [person for person in os.listdir("People/")]

	#loop through the folders checking for different people
	for i, person in enumerate(people):
		labels_dic[i] = person
		for image in os.listdir("People/" + person):
			images.append(cv2.imread("People/" + person + "/" + image, 0))
			labels.append(i)

	return (images, np.array(labels), labels_dic)

#Crops the face, so that 70% of the face is captured all the time
def cut_faces(frame, face_coord):
	faces = []

	for (x, y, w, h) in face_coord:
		w_rm = int(0.2 * w / 2)
		faces.append(frame[y: y + h, x + w_rm: x + w - w_rm])

	return faces

""" This is the main function that uses the Haar Cascade detector to detect 
	the faces and draw a rectangle aroung the edges of the face and return it
	to the main function
"""
def casDetector(frame, detector, biggest_only = True):
	#biggest_only = True
	scale_factor = 1.2
	min_neighbors = 5
	min_size = (30, 30)

	flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
    		cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
    		cv2.CASCADE_SCALE_IMAGE 

	face_coord = detector.detectMultiScale(
            frame,
            scaleFactor = scale_factor,
            minNeighbors = min_neighbors,
            minSize = min_size,
            flags = flags
        )

	return face_coord

#Normalizes the image to increase the contrast
def normalize_intensity(images):
	images_norm = []

	for image in images:
		is_color = len(image.shape) == 3
		if is_color:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		images_norm.append(cv2.equalizeHist(image))

	return images_norm

#Draw a rectangle around the face
def drawRectangle(face_coord):
	for (x, y, w, h) in face_coord:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 0), 2)
		cv2.line(frame, (x + (w/2), y), (x + (w/2), y - 20), (150, 150, 0), 2)

def talk(engine, text):
	engine.say("The person identified is")
	engine.say(text)
	engine.runAndWait()

#start of the main function
print("\n\n\t\tWelcome to the face recognition system!\n\n\t\tStarting to Train the data!")

#capture the video and create windows
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed for Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Feed for Recognition", 800, 400)

#saving the video recording
fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('Videos/videos2.avi', fourcc, 20.0, (640, 480))

#initialize the Haar Cascade
detector = cv2.CascadeClassifier("frontal_face.xml")

#collect the dataset from pre-trained models
images, labels, labels_dic = collect_dataset()

#train the collected data
rec_lbph = cv2.createLBPHFaceRecognizer()
rec_lbph.train(images, labels)
print("\n\n\t\tFinished Training!!")
font = cv2.FONT_HERSHEY_SIMPLEX

#Start time module
start_time = time.time()

#Initialize text to speech API
engine = pyttsx.init()
namecall = "Unknown"

#start of the image recognition function
while True:
		#reading the frame and saving the video
		ret, frame = webcam.read()
		video.write(frame)

		#obtaining the face coordinates using the function and start the loop to save images
		face_cord = casDetector(frame, detector, False)
		drawRectangle(face_cord)
		for (x, y, w, h) in face_cord:
			height = h

		if len(face_cord):
			#crop the images respectively acc to the face coords and convert to gray
			cut_img = cut_faces(frame, face_cord)

			#Normalize
			faces = normalize_intensity(cut_img)
			for i, face in enumerate(faces):
				pred, conf = rec_lbph.predict(face)
				pred_name = labels_dic[pred].capitalize()
				threshold = 55
				#print("Prediction-->" + pred_name + "\tConfidence-->" + str(conf) + "\n")
				if conf < threshold:
					cv2.putText(frame, pred_name, 
						(face_cord[i][0], face_cord[i][1] - 30), 
						font, .7, (66, 152, 243), 2, cv2.CV_AA)
					cv2.putText(frame, "Accuracy: " + str(round(conf)), 
						(face_cord[i][0], face_cord[i][1] + height + 20), 
						font, .5, (256, 100, 100), 2, cv2.CV_AA)
				else:
					cv2.putText(frame, "Unknown", 
						(face_cord[i][0], face_cord[i][1] - 30), 
						font, .9, (23, 202, 255), 2, cv2.CV_AA)

		cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (255, 0, 0), 1, cv2.CV_AA)
		cv2.imshow("Live Feed for Recognition", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("\n\n\t\tThanks for stopping by!\n\n\t\tTime Elaspsed is: " + str(time.time() - start_time))
			print("\n\n")
			break

cv2.destroyAllWindows()
webcam.release()
video.release()
