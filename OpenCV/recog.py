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

rec_eig = cv2.face.createEigenFaceRecognizer()
rec_eig.train(images, labels)

print("Trained Successfully!")