#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:22:54 2019

@author: anush
"""
#Importing libraries
import cv2
import numpy as np
import matplotlib as plt

#Loading the input image 
input_img = cv2.imread('../image.jpg', 0) 'the image is converted into grayscale'
height, width = input_img.shape

sobel_x = cv2.Sobel(input_img, cv2.CV_64F, 0, 1, ksize = 5) 'sobel_x captures horizontal edges'
sobel_y = cv2.Sobel(input_img, cv2.CV_64F, 1, 0, ksize = 5) 'sobel_y captures vertical edges'

sobel_or = cv2.bitwise_or(sobel_x, sobel_y) 'sobel_or combines both horizontal and vertical edges'
cv2.imshow('edges', sobel_or)
cv2.waitKey(0)
cv2.destroyAllWindows()

laplacian = cv2.Laplacian(input_img, cv2.CV_64F)
cv2.imshow('laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

canny = cv2.Canny(input_img, 40, 170)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
