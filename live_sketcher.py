"""Live Video Sketcher"""

'Importing Libraries'
import cv2
import numpy as np

'Defining a function that sketches the live image'
def sketch(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gray_image_blur = cv2.GaussianBlur(gray_image, (5,5), 0) 
    canny_edges = cv2.Canny(gray_image_blur, 10, 70) 
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV) 
    return mask

#Capturing the viseo
cap = cv2.VideoCapture(2) #index is 2 for the USB webcam
while True:
    ret, frame = cap.read()
    cv2.imshow('live_sketcher', sketch(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
