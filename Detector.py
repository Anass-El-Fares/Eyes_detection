# --------------- Eyes detection using Deep Learning -----------------
# ---------------------- El Fares Anass ------------------------------
# ---------------------- XX/XX/XXXX ----------------------------------

import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound_1 = mixer.Sound('Girls_Like_You.mp3')
sound_2 = mixer.Sound('Alarm.mp3')
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
labels =['Closed','Open']
model = load_model('models/CNN_eye.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
det = 0
rpred = [[1,0]]
lpred = [[1,0]]
try:
    sound_1.play()
except:
    pass
while(True):

    ret, frame = cap.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    cv2.rectangle(frame, (0, 430), (700, 480), (255, 50, 50), thickness=cv2.FILLED)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,100,0) , 1 )
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye= r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        if(rpred[0][0]>0.8):
            labels='Open'
        if(rpred[0][0]<0.2):
            labels='Closed'
        break
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        if(lpred[0][0]>0.8):
            labels ='Open'
        if(lpred[0][0]<0.2):
            labels = 'Closed'
        break

    testo = "Playing 'Girls like you' - Maroon 5"
    cv2.putText(frame, testo, (10, 20), font, 0.7, (100, 0, 100), 1, cv2.LINE_AA)

    if(rpred[0][0]>0.7 and lpred[0][0]>0.7):
        det = det + 1
        cv2.putText(frame,"Closed",(225,460), font, 1,(20,255,255),1,cv2.LINE_AA)
        print('Occhi CHIUSI !!')
    else:
        det = det - 5
        cv2.putText(frame,"Open",(225,460), font, 1,(255,255,255),1,cv2.LINE_AA)
        print('Occhi aperti ')

    if(det<0):
        det = 0
    cv2.putText(frame,'Detections:'+str(det),(325,460), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(det>3):
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        cv2.putText(frame, 'Allarme!', (30, 45), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        try:
            sound_1.set_volume(0.1)
            sound_2.play()
        except:
            pass
    else:
        sound_1.set_volume(1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
