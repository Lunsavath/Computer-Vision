#Importing libraries
import cv2
import numpy as np
import matplotlib as plt

#Reading the input image
image = cv2.imread('/home/image.jpg')

#Converting the image into grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Reducing the noise by applying canny edge filter
canny_edged = cv2.Canny(gray_image, 30, 200)

#Finding contours of the image
_, contours, hierarchy = cv2.findContours(canny_edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

