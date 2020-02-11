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

nmLen = 9   #tested, this is the most optimal length

import pandas as pd 

data = pd.read_csv("SSA_Names_DB.csv")
names = data.Name
gender = data.Gender

#converts the string names to integers
#since the program is computationally light, I am experiementing with increasing name length
num_names = [[0 for i in range(nmLen)] for j in range(len(data))]  #  j is nbr of names, i is length
names = np.array(names)

tmpName = [] #initializes array

for i in range(0,len(names)):   #this adds zero padding to the names and makes sure they are lower case
  if(len(names[i]) > nmLen):
    names[i] = names[i][:nmLen]
  names[i] = names[i].zfill(nmLen)
  names[i]= names[i].lower()
  
  for i in range(0,len(names)):
  for j in range (0,nmLen):
      num_names[i][j] = ord(names[i][j])
      num_names[i][j] = num_names[i][j] - 96 #Converts the ascii values to 0 through 27
      
  #converts the genders to integers
num_gender = [0 for i in range(len(gender))]
for i in range(len(gender)):
  if(gender[i] == "M"): #MALE
    num_gender[i] = 0
  if(gender[i] == "F"): #FEMALE
    num_gender[i] = 1
num_names = np.array(num_names) #converts num_names to a numpy array
num_names = (np.arange(num_names.max()) == num_names[...,None]).astype(int)  #converts num_names into one_hot encoded

import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

BATCH = 800            
L2_WEIGHT_DECAY = 2e-2 # experimented a few times with this value, seems this one is pretty good.


model = tf.keras.Sequential([

  tf.keras.layers.Conv1D(256, kernel_size= 8, input_shape = (nmLen,26,),          # Convolutional layer, using a kernel and bias,
                         use_bias = True, bias_initializer = 'random_uniform',    #bias is optiized using a weight decay regulizer.
                         bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY), ),

  tf.keras.layers.SeparableConv1D(256, kernel_size = 2, input_shape = (2,256),                   #this layer works the same as the prior, altough the convolution is in a
                        trainable = True,  use_bias = True, bias_initializer = 'random_uniform', #different vector direction, 
                         bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),),
  
  tf.keras.layers.Flatten(),    #returns to two dimensional nueral network, can only use fully connected now

  tf.keras.layers.LeakyReLU(alpha=0.15),    #gradient descent with leaky RELU, learning rate = alpha

  tf.keras.layers.Dense(128, activation = 'relu'), 
  tf.keras.layers.Dropout(rate = .25, noise_shape=(BATCH, 1) ),    #drop out layer, decreasing % increases learning speed, but decreases max accuracy.

  tf.keras.layers.Dense(64, activation = 'selu'), 
  tf.keras.layers.Dropout(rate = .3, noise_shape=(BATCH, 1) ), #selu was higher accuracy than relu

  tf.keras.layers.Dense(64, activation = 'relu'), 

  tf.keras.layers.Dense(32,  activation = 'relu'),

  tf.keras.layers.Dense(2, activation = 'softplus') #softplus is supperior to softmax for this algorithm
])

model.compile(optimizer='adam',                        #adam is the most optimal
              loss= 'sparse_categorical_crossentropy', #sparse_categorical_crossentropy is the most optimal
              metrics=['accuracy'])                    # accuracy is what we want to have

#uniformally shuffles the data, such that the gender and names still have their same assignments.
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
num_names = np.array(num_names)
num_gender = np.array(num_gender)
num_names, num_gender = shuffle_in_unison(num_names, num_gender)


model.fit(num_names[:100000],num_gender[:100000], batch_size = BATCH, epochs = 600, validation_data = (num_names[100000:], num_gender[100000:]), verbose = 1)


predict_guess = [ 'tyler' ]
namePred = [ 'tyler' ]

confLvl = .55

for i in range(0,len(predict_guess)):
  predict_guess[i] = predict_guess[i].zfill(nmLen)
  predict_guess[i]= predict_guess[i].lower()
  if(len(predict_guess[i]) > nmLen):
    predict_guess[i] = predict_guess[i][:nmLen]

num_predict = [[0 for i in range(nmLen)] for j in range(len(predict_guess))]

for i in range(0,len(predict_guess)):
  for j in range (0,nmLen):
      num_predict[i][j] = ord(predict_guess[i][j])
      num_predict[i][j] = num_predict[i][j] - 96 # EXP

num_predict = np.array(num_predict)
num_predict = (np.arange(26) == num_predict[...,None]).astype(int)

answer = model.predict(num_predict)

for i in range(0, len(namePred)):
  percentM = (answer[i][0] / (answer[i][0] + answer[i][1]) )
  percentF = (answer[i][1] / (answer[i][0] + answer[i][1]) )
  if(answer[i][0] > answer[i][1] and (percentM > confLvl)):
    print(namePred[i] + ": is a male, we are " + str(percentM) + "% confident.")
  if(answer[i][0] < answer[i][1] and percentF > confLvl):
    print(namePred[i] + ": is a female, we are " + str(percentF) + "% confident.")
  if(percentM < confLvl and percentF < confLvl):
    print(namePred[i] + " is gender nuetral - cannot tell")

   
