from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # Use the %tensorflow_version magic if in colab.
  %tensorflow_version 2.x
except Exception:
  pass

import random
import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd 

data = pd.read_csv("SSA_Names_DB.csv")
names = data.Name
gender = data.Gender

#converts the string names to integers
num_names = [[0 for i in range(10)] for j in range(len(data))] #  j is nbr of names, i is length
print(names[43])
for i in range(0,len(names)):
  names[i] = '{0:0<10}'.format(names[i])
  if(len(names[i]) > 10):
    names[i] = names[i][:10]

for i in range(0,len(names)):
  for j in range (0,10):
      num_names[i][j] = ord(names[i][j])
print(num_names[0])

#converts the genders to integers
num_gender = [0 for i in range(len(gender))]
for i in range(len(gender)):
  if(gender[i] == "M"):
    num_gender[i] = 0
  if(gender[i] == "F"):
    num_gender[i] = 1
    
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

model = tf.keras.Sequential([  # has accuarcy of ~71%
  tf.keras.layers.Dense(10, input_shape = (10,)), #guess work on this line
  tf.keras.layers.Dense(64, activation = 'tanh'),
  tf.keras.layers.Dense(2,  activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
num_names = np.array(num_names)
num_gender = np.array(num_gender)

model.fit(num_names,num_gender, epochs = 10)
