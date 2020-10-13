import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2
import dlib
import pyglet
import random

my_model = tf.keras.models.load_model('resources/my_model')
class_names = ['closed eye', 'open eye']

arena = np.zeros((600, 600, 3), np.uint8)
symbols_set = {0: "", 1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: ""}

sound_Ok = pyglet.media.load("resources/soundMy.wav", streaming=False)
sound_Veto = pyglet.media.load("resources/beep.wav", streaming=False)

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN
turn = "x"
frames = 0
index = 0
blinking_frames = 0
gaze_frames = 0
count = 0

img_width = 24
img_height = 24
dim = (img_width, img_height)

def draw_arena(index, symbol, cell_light, eye_blinking, turn_x_o):
    # Cells
    if index == 0:
        x = 0
        y = 0
    elif index == 1:
        x = 200
        y = 0
    elif index == 2:
        x = 400
        y = 0
    elif index == 3:
        x = 0
        y = 200
    elif index == 4:
        x = 200
        y = 200
    elif index == 5:
        x = 400
        y = 200
    elif index == 6:
        x = 0
        y = 400
    elif index == 7:
        x = 200
        y = 400
    elif index == 8:
        x = 400
        y = 400
    

    global count, turn
    
    # Cell settings
    width = 200
    height = 200
    th = 2 # thickness
    
    # Symbol settings
    font_symbol = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 6
    symbol_size = cv2.getTextSize(symbol, font_symbol, font_scale, font_th)[0]
    width_symbol, height_symbol = symbol_size[0], symbol_size[1]
    symbol_x = int((width - width_symbol) / 2) + x
    symbol_y = int((height + height_symbol) / 2) + y
    
    if cell_light is True:
        cv2.rectangle(arena, (x + th, y + th), (x + width - th, y + height - th), (174, 255, 0), -1)
        
        if turn_x_o == "x":
            if eye_blinking is True:
                if symbols_set[index] == "":
                    symbols_set[index] = turn_x_o
                    count += 1
                    turn = "o"
                    sound_Ok.play()
                else:
                    sound_Veto.play()
        else:
            computer_move(symbols_set)
    else:
        cv2.rectangle(arena, (x + th, y + th), (x + width - th, y + height - th), (148, 166, 0), th)
    
    if symbol == "x":
        cv2.putText(arena, symbol, (symbol_x, symbol_y), font_symbol, font_scale, (84, 84, 84), font_th)
    else:
        cv2.putText(arena, symbol, (symbol_x, symbol_y), font_symbol, font_scale, (243, 236, 208), font_th)

### ### ###

def someoneWon(symbols_set_win, turn_x_o, game_over):
    result = "No"
    if any([symbols_set_win[0] == symbols_set_win[1] == symbols_set_win[2] != "",
            symbols_set_win[3] == symbols_set_win[4] == symbols_set_win[5] != "",
            symbols_set_win[6] == symbols_set_win[7] == symbols_set_win[8] != "",
            symbols_set_win[0] == symbols_set_win[3] == symbols_set_win[6] != "",
            symbols_set_win[1] == symbols_set_win[4] == symbols_set_win[7] != "",
            symbols_set_win[2] == symbols_set_win[5] == symbols_set_win[8] != "",
            symbols_set_win[0] == symbols_set_win[4] == symbols_set_win[8] != "",
            symbols_set_win[2] == symbols_set_win[4] == symbols_set_win[6] != ""]):
        result = "Yes"
        if turn_x_o == "x":
            turn_x_o = "o"
        else:
            turn_x_o = "x"
        cv2.putText(arena, ("* " + turn_x_o + " *" + " WINNER!"), (100, 300), font2, 3, (255, 255, 255), thickness=3)
        cv2.destroyWindow("Eye") 
        cv2.destroyWindow("Frame") 
    elif game_over == len(symbols_set):
        result = "Tie"
        cv2.putText(arena, ("TIE!"), (250, 300), font2, 3, (255, 255, 255), thickness=3)
    return result

### ### ###

def computer_move(can_win):
    global count, turn
    can_win_coord = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
    for each in can_win_coord:
        if (can_win[each[0]] == can_win[each[1]] != "") and (can_win[each[2]] == ""):
            can_win[each[2]] = "o"
            count += 1
            turn = "x"
            break
        elif (can_win[each[0]] == can_win[each[2]] != "") and (can_win[each[1]] == ""):
            can_win[each[1]] = "o"
            count += 1
            turn = "x"
            break
        elif (can_win[each[1]] == can_win[each[2]] != "") and (can_win[each[0]] == ""):
            can_win[each[0]] = "o"
            count += 1
            turn = "x"
            break
    if turn == "o":
        moveComp = random.randint(0, 8)
        if symbols_set[moveComp] == "":
            symbols_set[moveComp] = "o"
            count += 1
            turn = "x"

### ### ###

while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    
    arena[:] = (174, 194, 0)

    blinking = False
    frames += 1
    
    faces = detector(frame) #use detector to find landmarks
    
    if (someoneWon(symbols_set, turn, count)) == "No":
        
        for face in faces:
            landmarks = predictor(frame, face)
            
            x1 = landmarks.part(36).x - 30 #top left
            x2 = landmarks.part(39).x + 30 #bottom right
            y1 = landmarks.part(36).y - 30 #top left
            y2 = landmarks.part(39).y + 30 #bottom right
            #cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
            
            right_eye_only = frame[y1: y2, x1: x2]
            #print (right_eye_only.size)
        
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
                blinking_frames += 1
                frames -= 1

                if blinking_frames == 3:
                    blinking = True

            else:
                blinking_frames = 0
                
            right_eye_only = cv2.resize(right_eye_only, None, fx=2, fy=2)
            cv2.imshow("Eye", right_eye_only)
                
        #Move to next cell
        if frames == 4: #delay
            index += 1
            frames = 0
        if index == 9:
            index = 0

        for i in range(9):
            if i == index:
                light = True
            else:
                light = False
            draw_arena(i, symbols_set[i], light, blinking, turn)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Tic-Tac-Toe", arena)

    key = cv2.waitKey(1)
    if key == 27: #esc
        break

cap.release()
cv2.destroyAllWindows()