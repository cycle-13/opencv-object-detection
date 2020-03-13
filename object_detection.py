import cv2
import numpy as np

SOURCE = 'data/qxdatasets/test_videos/walk_australia_372020.mp4'
capture = cv2.VideoCapture(SOURCE)

ret, frame1 = capture.read()
ret, frame2 = capture.read()

# KNOWN BUG: Program freezes if you try to exit the program. 
# Do not worry, this only applies if you are using Jupyter Notebook. 
# Just code in your IDE and everything will be fine.

while capture.isOpened():
    difference = cv2.absdiff(frame1, frame2)
    grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5,5), 0)
    _, threshold = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    dilated =  cv2.dilate(threshold, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1300:
            continue
        cv2.rectangle(frame1, (x,y), (x+w,y+h),(255,0,255), 2) 
        cv2.putText(frame1, 'STATUS: {}'.format('MOVEMENT!'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,
                   1,(0,255,0), 2)
    cv2.imshow('video', frame1)
    frame1 = frame2
    ret, frame2 = capture.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

