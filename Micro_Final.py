#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import dlib
from math import hypot
import time


# In[2]:


# For detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[15]:


# For Keyboard Setup
keyboard = np.zeros((500, 1500, 3),np.uint8)

keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T", 5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G", 15: "H", 16: "J", 17: "K", 18: "L", 19: ".",
    20: "Z", 21: "X", 22: "C", 23: "V", 24: "B", 25: "#", 26: "N", 27: "M", 28: "?", 29: "!"}


# In[9]:


def letter(letter_index, text, letter_light):
    width = 150
    height = 150
    th = 3  # thickness

    # Calculate row and column indices
    row = letter_index // 10
    col = letter_index % 10

    x = col * width
    y = row * height

    if letter_light:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 5
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)


# In[5]:


def midpoint(point1, point2):
    return int((point1.x + point2.x)/2), int((point1.y + point2.y)/2)


# In[6]:


def get_blinking_ratio(eye_point, facial_landmarks):
    left_point = (facial_landmarks.part(eye_point[0]).x, facial_landmarks.part(eye_point[0]).y)
    right_point = (facial_landmarks.part(eye_point[3]).x, facial_landmarks.part(eye_point[3]).y)
    
    center_top = midpoint(facial_landmarks.part(eye_point[1]), facial_landmarks.part(eye_point[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_point[5]), facial_landmarks.part(eye_point[4]))
    
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255,0), 1)
        
    hor_line_length = hypot ((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot ((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
    if ver_line_length == 0:
        return 0.0
    
    ratio = hor_line_length / ver_line_length
    return ratio


# In[7]:


def get_gaze_ratio(eye_point, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_point[0]).x, facial_landmarks.part(eye_point[0]).y),
                                    (facial_landmarks.part(eye_point[1]).x, facial_landmarks.part(eye_point[1]).y),
                                    (facial_landmarks.part(eye_point[2]).x, facial_landmarks.part(eye_point[2]).y),
                                    (facial_landmarks.part(eye_point[3]).x, facial_landmarks.part(eye_point[3]).y),
                                    (facial_landmarks.part(eye_point[4]).x, facial_landmarks.part(eye_point[4]).y),
                                    (facial_landmarks.part(eye_point[5]).x, facial_landmarks.part(eye_point[5]).y)], np.int32)
    
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
        
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
        
    gray_eye = eye[min_y: max_y, min_x: max_x]
    
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    
    if threshold_eye is None:
        print("Thresholding failed or no eye detected.")
        return 0 
    
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
        
    right_side_threshold = threshold_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    
    return gaze_ratio


# In[17]:


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

board = np.zeros((500, 500), np.uint8)
board[:] = 255

frames = 0
blinking_frames = 0
letter_index = 0

last_gaze_change_time = time.time() 
delay_duration = 1.5

text = ""
x_offset = 10

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    keyboard[:] = (0, 0, 0)
    frames += 1
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = ""
    
    active_letter = keys_set_1[letter_index]
    
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0, 255,0), 1)
        
        landmarks = predictor(gray, face)
        
        # Detect Blinking
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        if blinking_ratio > 6:
            cv2.putText(frame, "Blinking", (50,150), font, 1, (255,0,0))
            blinking_frames += 1
            frames -= 1
            
            if blinking_frames == 5:
                text += active_letter
        else:
            blinking_frames = 0
            
        # Gaze Detection
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_left_eye) / 2
        
        if gaze_ratio < 1:
            cv2.putText(frame, "LEFT", (50,100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0,0,255)
        elif 1 < gaze_ratio < 2:
            cv2.putText(frame, "CENTER", (50,100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "RIGHT", (50,100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (255,0,0)
        
        
        # Letters
        current_time = time.time()  # Get current time
        if gaze_ratio < 1 and current_time - last_gaze_change_time > delay_duration:
            last_gaze_change_time = current_time  # Update last gaze change time
            letter_index -= 1
            if letter_index < 0:
                letter_index = 29
        elif gaze_ratio > 2 and current_time - last_gaze_change_time > delay_duration:
            last_gaze_change_time = current_time  # Update last gaze change time
            letter_index += 1
            if letter_index > 29:
                letter_index = 0

        for i in range(30):
            if i == letter_index:
                light = True
            else:
                light = False
            letter(i, keys_set_1[i], light)
            
        for texts in text:
            cv2.putText(board, texts, (x_offset, 50), font, 2, 0, 3)
            x_offset += 45
        
        
    #cv2.imshow("Frame", frame)
    #cv2.imshow("New Frame", new_frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




