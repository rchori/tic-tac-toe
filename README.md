# Сlassic Tic-Tac-Toe game in an unconventional way


**resources** folder contains media files, link to a trained facial shape predictor and the built classification model (open/closed eye), the dataset of a right eye with two states (open and closed)

**open_closed_eye_detection.py** detects eye frame based on face_landmark_detection and identify whether the right eye is open or closed with the help of the classification model

**model.py** builds image classification model using Convolutional Neural Networks

Note: the appropriate distance between a web camera and an eye is about 30-40 cm
