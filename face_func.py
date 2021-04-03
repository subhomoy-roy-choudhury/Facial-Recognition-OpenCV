# -*- coding: utf-8 -*-
"""
@author: Subhomoy.Roy.Choudhury
"""
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from shot import shot



# from PIL import ImageGrab

path = 'Images_train'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')


def face():

    shot('face.png')
    img = cv2.imread("opencv.png")
    # success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            label_position = (150,200)
            cv2.putText(img,name,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imwrite("face_test.png",img)
            return name
            # return 'unlocked'
        else:
            print('not found')

            


# if __name__ == '__main__':
#     while True:
#         face() 

