import os
import numpy as np
import cv2
import pickle
import face_recognition
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

names = []
encodings = []
knownEncodings =[]
knownNames = []
#if any names and encodings were previously gathered, open them
if(os.path.exists("names.pickle")):
    with open("names.pickle",'rb') as nameFile:
        knownNames = pickle.load(nameFile)
if(os.path.exists("enc.pickle")):
    with open("enc.pickle",'rb') as encFile:
        knownEncodings = pickle.load(encFile)

while True:
    name = input("Enter the name of the person in the video feed: ")
    name = name.upper()
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    captures = 0
    frames = []
    startTime = time.time()

    #save the images from the webcam feed until 10s elapse
    while cap.isOpened():
        ret , frame = cap.read()

        if ret:

            frames.append(frame)

            elapsedTime = round(time.time() - startTime,2)
            cv2.putText(frame,str(elapsedTime),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

            cv2.imshow(name, frame)

            if(elapsedTime >= 10):
                break
            cv2.waitKey(1)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Extracted ',len(frames), 'frames of ', name)
    print('Extracting 128d embeddings...')

    #iterate over the images saved and generate the encodings for them
    for img in frames:
        imgSmall = cv2.resize(img,(0,0),None,0.20,0.20)
        facesInFrame = face_recognition.face_locations(img)
        encodingsInFrame = face_recognition.face_encodings(img,facesInFrame)

        if facesInFrame:
            names.append(name)
            encodings.append(encodingsInFrame[0])
            captures += 1
    print('Of ', len(frames), ' frames, extracted ',captures,' valid frames')

    cont = input('Would you like to add another person? (Y/N): ')
    if cont.upper() == 'N':
        break

#append the existing data to the knewly collected data
encodings = encodings + knownEncodings
names = names + knownNames

if(len(encodings) > 0):
    print('Total encodings: ',len(encodings))

    print('Training Model...')
    le = LabelEncoder()
    labels = le.fit_transform(names)

    #ensure there is more than 1 person in the dataset
    if(len(le.classes_) <= 1):
        print('Need to train the model on at least two people. Please try again.')

    else:
        #training the SVM
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(encodings, labels)
        print('Training Complete')

        #save the data for the recognizer module to access
        with open("names.pickle",'wb') as nameFile:
            pickle.dump(names,nameFile)


        with open("enc.pickle",'wb') as encFile:
            pickle.dump(encodings,encFile)

        with open("le.pickle",'wb') as leFile:
            pickle.dump(le,leFile)


        with open("rec.pickle",'wb') as recFile:
            pickle.dump(recognizer,recFile)
