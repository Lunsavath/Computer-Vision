# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, Dropout, MaxPool2D, Flatten

#Creating path to DATADIR
DATADIR = "/home/anush/work/dataset/"
CATEGORIES = ["dogs", "cats"]

#Mentioning image size
IMG_SIZE = 100

#Creating training data

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

#Shuffling the training data
random.shuffle(training_data)

#Creating labels and features
X = []
y = []

for features,labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Normalizing the images
X = X/255.0

#Building the model
model = Sequential()

#Layer 1
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='elu', kernel_initializer='glorot_uniform', input_shape = X.shape[1:]))
model.add(MaxPool2D(pool_size=(2,2)))

#layer 2
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPool2D(pool_size=(2,2)))

#Layer 3
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPool2D(pool_size=(2,2)))

#Layer 4
model.add(Flatten())
model.add(Dense(64))

#Layer 5
model.add(Dense(1))
model.add(Activation("sigmoid"))

#Compiling the data
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Fitting the data
model.fit(X, y, batch_size = 50, epochs= 500, validation_split= 0.1)


