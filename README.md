# Face Recognition Project

This Git Repository is a collection of various papers and code on the face recognition system using **Python2.7** and **OpenCV 2.4.8**.
#### The folder *OpenCV* contains the complete files that is related to compiling and running the Face Recognition System.
The .pdf files in this repo are some of the earliest and the fundamental papers on this topic. It also includes the Research paper from Google's *FaceNet* and Taigman's *DeepFace*.

# Dependencies
* numpy
* opencv
* matplotlib
* time
* math
* os
* pyttsx (For TexttoSpeech)
Use pip to install any missing dependencies

# Usage
There are two files namely `Pydata1.py` and `Pydata2.py`, which are two main python codes needed to run the system.

The `Pydata1.py` contains the code for **Building the dataset for our image recognition model**. It captures 400 Images of the person who needs to be recognised. This images are stored in the [People](/OpenCV/People) Folder in the OpenCV Folder.

After running the `Pydata1.py` in the terminal, all the images are saved and we are now ready to run the other file `Pydata2.py`.

This is the main face recognition code, which can be used to recognize the pre-trained images or flag as unknown if the image is not recognized.

The training code is:
```python
rec_lbph = cv2.createLBPHFaceRecognizer()
rec_lbph.train(images, labels)
```
This trains the images using a **LBPH** (Linear Binary Pattern Histogram) model and checks if the images are recognizable under a set threshold of 70. It also predicts the confidence and labels the image recognized accordingly.

The final results are somewhat like this (this is a picture of me and my mom, which I have pre-trained to but the other picture i haven't).
<img width="1440" alt="screen shot 2017-08-14 at 9 11 16 pm" src="https://user-images.githubusercontent.com/12717969/29313786-0334eaee-81d9-11e7-97cc-311cbf65367e.png">