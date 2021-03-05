"""
Project: UTMIST 2020-2021 Hand Gesture Recognition Project
Created on: Fri March 5, 2021
@author: Mustafa Khan
File description: Initializing trained c3d_superlite model to run inferences
Requirements: tensorflow, keras, opencv, numpy + the model weights
"""
import numpy as np
import cv2
import tensorflow as tf
import os
import math
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input,Flatten,Dense,Dropout,Activation,Reshape
from keras.layers import Conv2D
from keras.layers import Conv3D,MaxPooling3D,ZeroPadding3D,MaxPool3D
from keras.layers import LeakyReLU,ReLU,Lambda
from keras.optimizers import Adam,SGD
from keras.layers import LSTM
from keras.regularizers import l2
from keras.applications.mobilenet_v2 import MobileNetV2

###############################################################

# the classes we want to use
targets_name = ['Swiping Up', 
                'Swiping Down', 
                'Swiping Left',
                'Swiping Right',
                'No gesture',
                'Doing other things']

def c3d_super_lite():
    """
    Lite C3D Model + LSTM
    L2 Normalisation of C3D Lite Feature Vectors 
    3M Parameters
    Able to Run on the Jetson Nano at 8FPS
    Final Dense Layer to bemodified to adjust the number of classes
    """
    shape = (30,112,112,1)

    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=shape))
    model.add(keras.layers.Conv3D(32, 3,strides=(1,2,2), activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(keras.layers.Conv3D(64, 3, activation='relu', padding='same', name='conv2'))
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='conv3a'))
    model.add(keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='conv3b'))
    model.add(keras.layers.MaxPooling3D(pool_size=(3,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='conv4a'))
    model.add(keras.layers.Conv3D(128, 3, activation='relu', padding='same', name='conv4b'))
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

    model.add(keras.layers.Reshape((9,384)))

    #model.add(keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    model.add(keras.layers.Lambda(lambda x: tf.math.l2_normalize(x,axis=-1)))
    
    model.add(keras.layers.LSTM(512, return_sequences=False,
                   input_shape= (9,384),
                   dropout=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(6, activation='softmax')) #7 was previously 6

    model.compile(loss='sparse_categorical_crossentropy', #previously categorical_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    return model 
    
model = c3d_super_lite()
# choose the loss and optimizer methods
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

#path to checkpoint
checkpoint_path = "./c3d_superlite_best_checkpoint.ckpt" #change this path
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
model.load_weights(checkpoint_path)

#%%
###############################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def normaliz_data(np_data):
    # Normalisation
    scaler = StandardScaler()
    #scaled_images  = normaliz_data2(np_data)
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images

def normaliz_data2(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v

###############################################################
to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))
    
         
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
        predict = new_model.predict(frame_to_predict)
        classe = classes[np.argmax(predict)]
        
        print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')


        #print(frame_to_predict)
        to_predict = []
        #sleep(0.1) # Time in seconds
        #font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()