#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


#import keras library
import keras


# In[3]:


#import ImageDataGenertator class from keras
from keras. preprocessing.image import ImageDataGenerator


# In[24]:


#define the parameter /arguments for ImageDataGenrator class
train_datagen=ImageDataGenerator(rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)

test_data=ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
# In[55]:


#: Applying ImageDataGenerator functionaly to trainset
x_train=train_datagen.flow_from_directory(r"C:\Users\822619104001.AITCS\Downloads\Dataset\training_set",target_size=(64,64),batch_size=300,class_mode= 'categorical',color_mode = "grayscale")


# In[ ]:





# In[56]:


#: Applying ImageDataGenerator functionaly to testset
x_test = test_datagen.flow_from_directory(r"C:\Users\822619104001.AITCS\Downloads\Dataset\test_set",
                                         target_size=(64,64),
                                         batch_size=300,
                                         class_mode= 'categorical',
                                         color_mode = "grayscale")


# In[ ]:





# In[ ]:





# In[ ]:





# # Model building

# In[35]:


#import imagedatagenerator
from keras.preprocessing.image import ImageDataGenerator


# In[39]:


#traning datagen
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True)


# In[40]:


#testing datagen
test_datagen = ImageDataGenerator(rescale=1./255)


# In[41]:


import tensorflow as tf
import os


# # initialize the model

# In[45]:


#create model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Maxpooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[46]:


import numpy as np
import matplotlib.pyplot as plt #to view graphs in colob itself
import IPython.display as display
from PIL import Image
import pathlib


# # Unzipping the dataset

# In[2]:





# In[ ]:




