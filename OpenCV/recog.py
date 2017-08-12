import os
import numpy as np 
import math
import time
from matplotlib import pyplot as plt 
import cv2

def collect_dataset():
	images = []
	labels = []
	labels_dic = {}
	people = [person for person in os.listdir("People/")]

	for i, person in enumerate(people):
		labels_dic[i] = person
		for image in os.listdir("People/" + person):
			images.append(cv2.imread("People/" + person + "/" + image, 0))
			labels.append(i)

	return (images, np.array(labels), labels_dic)

images, labels, labels_dic = collect_dataset()

rec_eig = cv2.createEigenFaceRecognizer()
rec_eig.train(images, labels)

rec_fisher = cv2.createFisherFaceRecognizer()
rec_fisher.train(images, labels)

rec_lbph = cv2.createLBPHFaceRecognizer()
rec_lbph.train(images, labels)

print("Trained Successfully!")

face = cv2.imread("7.jpg", cv2.COLOR_BGR2GRAY)

pred, conf = rec_eig.predict(face)
print("Eigen Prediction-->" + labels_dic[pred].capitalize() + "\tConfidence-->" + str(conf) + "\n")

pred, conf = rec_fisher.predict(face)
print("Fisher Prediction-->" + labels_dic[pred].capitalize() + "\tConfidence-->" + str(conf) + "\n")

pred, conf = rec_lbph.predict(face)
print("LBPH Prediction-->" + labels_dic[pred].capitalize() + "\tConfidence-->" + str(conf) + "\n")