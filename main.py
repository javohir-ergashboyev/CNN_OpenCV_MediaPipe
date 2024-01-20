from tensorflow.keras.models import load_model
import cv2 as cv
import mediapipe as mp
import numpy as np

X=np.load('book.npy')
x_res=cv.resize(X, (28,28))
cv.imshow('t',x_res)
X/255
print(X.shape)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()

model=load_model('doodle_model.h5')

def return_str(n):
    if n==0:
        return 'an apple'
    elif n==1:
        return 'a banana'
    elif n==2:
        return 'a basket'
    elif n==3:
        return 'a book'
    else: return 'Not Detected'
cap=cv.VideoCapture(1)
prev_x,prev_y=0,0
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
while cap.isOpened:
    _, frame=cap.read()
    frame=cv.flip(frame,1)
    frame_rgb=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result=hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_coords = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_cords=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w= frame.shape[:2]
            x, y = int(index_finger_coords.x * w), int(index_finger_coords.y * h)
            x1,y1=int(middle_finger_cords.x * w), int(middle_finger_cords.y * h)
            x2,y2=int(thumb.x * w), int(thumb.y * h)
            distance=((((x1-x)**2+(y1-y)**2)**0.5))
            distance2=((((x2-x)**2+(y2-y)**2)**0.5))
            distance3=((((x2-x1)**2+(y2-y1)**2)**0.5))
            if distance2>30:
                if distance<=30: 
                    cv.line(canvas, (x,y), (x, y), (0, 0, 0), 25)
                    cv.circle(frame,(int((x+x1)/2), int((y1+y)/2)), int(distance),(0, 255, 0), -1)
                else:
                    prev_x, prev_y = x, y
                    cv.line(canvas, (x,y), (prev_x, prev_y), (0, 255, 0), 25)
                    cv.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    prev_x, prev_y = x, y
            if distance3<10:
                canvas=np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        if np.any(canvas!=0):
            c=cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
            resized=cv.resize(c, (28,28))
            canvas_processed = np.reshape(resized,(784,))
            normalized=canvas_processed/255
            predict=model.predict(np.expand_dims(normalized, axis=0))
            res=np.argmax(predict)
            cv.imshow('resized',resized)
            cv.putText(frame,f'You draw {return_str(res)}', (10,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

    imgGray=cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, inv=cv.threshold(imgGray, 50,255, cv.THRESH_BINARY_INV)
    inv=cv.cvtColor(inv, cv.COLOR_GRAY2BGR)
    frame=cv.bitwise_and(frame, inv)
    frame=cv.bitwise_or(frame, canvas)
    
    cv.imshow('Frame',frame)
    if cv.waitKey(10) & 0xff==ord('q'):
        break
cap.release()
cv.destroyAllWindows()


