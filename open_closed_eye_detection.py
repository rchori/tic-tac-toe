import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import dlib
import pyglet

my_model = tf.keras.models.load_model('resources/my_model')
class_names = ['closed eye', 'open eye']

cap = cv2.VideoCapture(0)
#sound = pyglet.media.load("resources/soundMy.wav", streaming=False)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()
    
    img_width = 24
    img_height = 24
    dim = (img_width, img_height)
    
    faces = detector(frame)
    for face in faces:
        
        #Eye frame detection
        landmarks = predictor(frame, face)
        
        x1 = landmarks.part(36).x - 30
        x2 = landmarks.part(39).x + 30
        y1 = landmarks.part(36).y - 30
        y2 = landmarks.part(39).y + 30
        #cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
        
        right_eye_only = frame[y1: y2, x1: x2]
        ###
        
        #Blink detection
        img = cv2.resize(right_eye_only, dim)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = my_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        res_pred = ("{} - {:.2f} %".format(class_names[np.argmax(score)], 100 * np.max(score)))

        cv2.putText(frame, res_pred, (50, 100), font, 1, (255, 255, 255), thickness=3)
        ###
        
        if class_names[np.argmax(score)] == 'closed eye':
            cv2.putText(frame, "BLINKING", (50, 200), font, 1, (255, 255, 255), thickness=3)
            #sound.play()

        right_eye_only = cv2.resize(right_eye_only, None, fx=2, fy=2)
        cv2.imshow("Eye", right_eye_only)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: #esc
        break

cap.release()
cv2.destroyAllWindows()
