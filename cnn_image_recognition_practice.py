#!/usr/bin/env python
# coding: utf-8

# In[1]:


#img recognition (binary classification problem)-CNN(Convolutional neural network)
#1st-full_connection: relu
#output layer: sigmoid


# In[2]:


#import the libraries


# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


#----------data processing 


# In[5]:


#processing the TRAINING set 
#1. generaete the traing_data
train_data_generator = ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.2,
                                          zoom_range = 0.2,
                                          horizontal_flip = True)
#2. import the training img_set(target_size, batch...)
training_set = train_data_generator.flow_from_directory('dataset/training_set',
                                                         target_size = (64, 64),
                                                         batch_size = 32,
                                                         class_mode = 'binary')


# In[6]:


#processing the TEST set 
#1. generaete the test_data
test_data_generator = ImageDataGenerator(rescale = 1./255)
#2. import the test img_set(target_size, batch...)
test_set = test_data_generator.flow_from_directory('dataset/test_set',
                                                         target_size = (64, 64),
                                                         batch_size = 32,
                                                         class_mode = 'binary')


# In[7]:


#----------building the model(CNN)


# In[8]:


#initializing CNN
cnn = tf.keras.models.Sequential()


# In[9]:


#1st layer
#1.convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
#2.pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[10]:


#2nd layer
#1.convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
#2.pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[11]:


#flattening (make the data colum)
cnn.add(tf.keras.layers.Flatten())


# In[12]:


#---neural network starts---


# In[13]:


#full connection(dense)
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[14]:


#output layer(dense)
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[15]:


#----------training the model(CNN)


# In[16]:


#compiling CNN(optimizer, loss, metrics)
cnn.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[17]:


#fit the model(CNN) on training_set and evaluate the test_set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[18]:


#----------making a single prediction
#(img size=target_size always haves to be the same!)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
#make 1 or 0
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

