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

for i in range(0,len(names)-1):
  names[i] = '{0:0<9}'.format(names[i])
  if(len(names[i]) > 10):
    names[i] = names[i][:9]
