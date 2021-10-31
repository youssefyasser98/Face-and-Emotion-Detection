# -*- coding: utf-8 -*-
"""
Created on Fri May 29 02:52:52 2020

@author: moda9
"""
import os
from imutils.video import FPS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import win32com.client
import cv2
import face_recognition
import numpy as np


process_this_frame = True

def face_compare(frame,process_this_frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    return face_names

Known_faces_dir = "dataset"
known_face_encodings = []
known_face_names = []

for name in os.listdir(Known_faces_dir):
    for filename in os.listdir(f"{Known_faces_dir}/{name}"):
        image = face_recognition.load_image_file(f"{Known_faces_dir}/{name}/{filename}")
        try:
            encoding = face_recognition.face_encodings(image)[0]
        except:
            break
        known_face_encodings.append(encoding)
        known_face_names.append(name)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('Emotion.h5')

class_labels = ['Angry','Happy','Neutral','Sad']

cap = cv2.VideoCapture(0)
fps = FPS().start()

speaker = win32com.client.Dispatch("SAPI.SpVoice")
# cap.set(3, 640)
# cap.set(4, 480)
#cap.set(cv2.CAP_PROP_FPS,10)
ret, frame = cap.read()
rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_name = face_compare(rgb_image,process_this_frame)
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #cap_fps = cap.get(5)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #face_name = "none"
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    k = cv2.waitKey(1)
    label_text="None"
    if not process_this_frame:
        face_name = face_compare(rgb_image,process_this_frame)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        #Rescaling of image
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            if face_name== "Unknown":
                label_text=label
            else:
                try:
                    label_text=f"{face_name[0]} is {label}"
                except IndexError:
                    label_text=label
            cv2.putText(frame,label_text,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
           
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    if k%256 == 32:
        speaker.Speak(label_text)
                # tts = gTTS(text=label, lang='en')
                # tts.save("1.mp3")
                # playsound('1.mp3')
    # cv2.resizeWindow('Emotion Detector',1280,720)
    cv2.imshow('See As We See',frame)
    fps.update()
    fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # print("[INFO] Cap. FPS: {:.2f}".format(cap_fps))
    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
