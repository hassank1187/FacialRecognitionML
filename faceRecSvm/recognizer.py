import cv2
import pickle
import os
import face_recognition
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg
import pynput

#pressing any key stops the alarm
def on_press(key):
    if pg.mixer.music.get_busy():
        pg.mixer.music.stop()

def on_release(key):
    pass

pg.mixer.init(44100, -16,2,2048)

# Collect events until released
listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

#open the model and label encoder
with open("le.pickle",'rb') as leFile:
    le = pickle.load(leFile)

with open("rec.pickle",'rb') as recFile:
    rec = pickle.load(recFile)

cap = cv2.VideoCapture(0)

successiveNoMatch = 0
while cap.isOpened():
    ret , img = cap.read()

    if ret:
        #resize the image to make it faster to process
        imgSmall = cv2.resize(img,(0,0),None,0.20,0.20)
        imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

        facesInFrame = face_recognition.face_locations(imgSmall)
        encodingsInFrame = face_recognition.face_encodings(imgSmall,facesInFrame)

        noMatchExists = False
        #compare each face in the frame against the SVM to find a match, if any
        for currEncoding , faceLocation in zip(encodingsInFrame, facesInFrame):

            predVec = []
            predVec.append(currEncoding)
            prediction = rec.predict_proba(predVec)[0]
            matchIndex = np.argmax(prediction)
            prob = prediction[matchIndex]
            recName = le.classes_[matchIndex]

            #35% match is the threshold for positive results
            matchFound = prob > 0.7
            color = (0,255,0)
            y1,x2,y2,x1 = faceLocation
            y1,x2,y2,x1 = y1 * 5, x2 * 5, y2 * 5, x1 * 5

            noMatchRois = []

            if matchFound:
                name = recName.upper()
            else:
                name = 'NO MATCH'
                color = (0,0,255)
                noMatchExists = True
                noMatchRois.append(img[y1:y2,x1:x2])


            #if there is anyone in frame not seen then its a no match
            if noMatchExists:
                successiveNoMatch += 1
            else:
                successiveNoMatch = 0

            #4 frames of no match are required to ensure no false negative
            if successiveNoMatch == 4:
                #play the alarm and extract the ROI of the individual(s)
                if not pg.mixer.music.get_busy():
                    pg.mixer.music.load('Air-raid-siren.mp3')
                    pg.mixer.music.play(loops = -1)

                for roi in noMatchRois:
                    txt = 'Unrecognized, Probability: ' + str(prob)
                    cv2.imshow(txt,roi)


            #putting the text and detection box on the frame
            textLocation = (x1,y1)
            cv2.putText(img,name,textLocation,cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)



        if not facesInFrame:
            successiveNoMatch = 0
            text = 'NO FACE DETECTED'
            cv2.putText(img,text,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)




        cv2.imshow('Webcam Feed', img)

        #press q to stop
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
    else:
        break

pg.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
