"""Template Matching"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/home/image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('/home/template_image.jpg', 0)
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0]+50, top_left[1]+50)
cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
cv2.imshow('matching', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
