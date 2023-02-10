'''
Created on 08.02.2023
@author: Devraj Pahariya
'''
import cv2 
import numpy
cas_path = "dev.xml"
face_Cascade = cv2.CascadeClassifier(cas_path)
video_cap = cv2.VideoCapture(0)
while True:
    ret, frame_= video_cap.read()
    gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    faces_ = face_Cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    for (x, y, w, h) in faces_:
       cv2.rectangle(frame_, (x, y), (x+w, y+h), (0, 255, 0), 2)
       cv2.putText(frame_,"Face", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,200,200), 2)
       cv2.putText(frame_,str(len(faces_)), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,200,200), 2)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found_Locations_Of_Body, foundWeights = hog.detectMultiScale(
        frame_,
        winStride=(8, 8), 
        padding=(32, 32), 
        scale=1.05
    )
    for x, y, w, h in found_Locations_Of_Body:
        if len (found_Locations_Of_Body) > 0:
            cv2.rectangle(frame_, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow("Stylebox", cv2.WND_PROP_FULLSCREEN)    
    cv2.setWindowProperty("Stylebox", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Stylebox", frame_)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video_cap.release()
cv2.destroyAllWindows()
