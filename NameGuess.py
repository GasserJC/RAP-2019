from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # Use the %tensorflow_version magic if in colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd 

data = pd.read_csv("SSA_Names_DB.csv")
names = data.Name
gender = data.Gender

# converts the string names to integers
num_names = [[0 for i in range(10)] for j in range(len(data))]  # j is nbr of names, i is length

for i in range(0,len(names)): # makes all strings same length
    names[i] = '{0:0<9}'.format(names[i])
    if len(names[i]) > 10:
        names[i] = names[i][:9]

# makes all strings a list of ascii values and inputs to a new 2d array
for i in range(0,len(names)):
    for j in range(0, 9):
        num_names[i][j] = ord(names[i][j])

# converts the genders to integers
for i in range(len(gender)):
    if i == 'M':
        i = 0
    else:
        i = 1
