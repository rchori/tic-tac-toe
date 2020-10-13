# Сlassic Tic-Tac-Toe game in an unconventional way


**resources** folder contains media files, link to a trained facial shape predictor and the built classification model (open/closed eye), the dataset of a right eye with two states (open and closed)

**game.py** runs Tic-Tac-Toe Game

**model.py** builds the image classification model using Convolutional Neural Networks

**open_closed_eye_detection.py** detects eye frame based on face_landmark_detection and identify whether the right eye is open or closed with the help of the classification model

Note: the appropriate distance between a web camera and an eye is about 30-40 cm

# How to play

1. Download game.py script and resources folder on to your computer (make sure to download shape_predictor_68_face_landmarks via the link provided)
1. Make sure to install the needed libraries (tensorflow, cv2, cmake, dlib, pyglet)
1. Run game.py by python of the 3rd version

As a result, given your right eye is detected, you will see 3 windows: the playing field, the web camera stream, your right eye

# Game logic

* 1
* 2
* 3
