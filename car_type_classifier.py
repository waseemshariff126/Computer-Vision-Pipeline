# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:03:52 2020

@author: wasee
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image

def car_type_model():
    base_model = tf.keras.applications.MobileNetV2(include_top = False, weights='imagenet', input_shape=(180, 180, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        Dense(512,activation='relu'),
        Dense(512,activation='relu'),
        Dense(256,activation='relu'),
        Flatten(),
        Dense(1, activation ='sigmoid')])
        
    model.compile(optimizer=RMSprop(lr = 0.001),loss='binary_crossentropy',metrics=['accuracy'])
    
    train_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                     validation_split=0.2)
    train_generator=train_datagen.flow_from_directory('C:/Users/wasee/New folder/Waseem_Moiz_ResearchTopic2/Training',
                                                     target_size=(180, 180),
                                                     color_mode='rgb',
                                                     batch_size=64,
                                                     class_mode='binary',
                                                     shuffle=True)
    validation_generator = train_datagen.flow_from_directory(
                                                    'C:/Users/wasee/New folder/Waseem_Moiz_ResearchTopic2/Training', # same directory as training data
                                                    target_size=(180,180),
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    subset='validation',
                                                    shuffle=True) # set as validation data
    
    step_size_train=train_generator.n//train_generator.batch_size
    step_size_val = validation_generator.samples // validation_generator.batch_size
    
    #    print(step_size_train)
    #    print(train_generator.n)
    #    print(train_generator.batch_size)
    
    model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                        validation_data = validation_generator,
                        validation_steps =step_size_val,
                       epochs=5)
    #return model
    tf.saved_model.save(model,'callback_model')
    




    
    
    
    
    
    
    