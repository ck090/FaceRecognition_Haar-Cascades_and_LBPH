print("Welcome")
folder = "People/" + raw_input('Enter the Persons Name: ').lower()
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live  Feed", 800, 400)
detector = cv2.CascadeClassifier("frontal_face.xml")

if not os.path.exists(folder):
	os.mkdir(folder)
	timer = 0
	counter = 0
	while counter < 10:
		ret, frame = webcam.read()
		face_cord = drawRect(frame, detector)
		if len(face_cord) and timer % 700 == 50:
			cut_img = cut_faces(frame, face_cord)
			gray = cv2.cvtColor(cut_img[0], cv2.COLOR_BGR2GRAY)
			norm_face = cv2.equalizeHist(gray)
			cv2.imwrite(folder + '/' + str(counter) + '.jpg', norm_face[0])
			plt_show(faces[0], "Image Saved: " + str(counter))
			counter = counter + 1
			resized_face = cv2.resize(norm_face, (50, 50), interpolation = cv2.INTER_AREA)
			#resized_face = resize(norm_face)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (0,0,0), 1, cv2.CV_AA)
		cv2.imshow("Live Feed", frame)
		cv2.waitKey(50)
		timer += 50
	cv2.destroyAllWindows()
else:
	print("This name exists")