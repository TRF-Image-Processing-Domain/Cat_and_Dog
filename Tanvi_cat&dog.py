#!/usr/bin/env python
# coding: utf-8



"""------------Data preprocessing---------------"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm




DATADIR = "D:/TRF/Image_Processing_TRF/Dogs_and_Cats/PetImages" #loading our dataset

CATEGORIES = ["Dog", "Cat"]
training_data =[]


# In[ ]:


IMG_SIZE = 100  #resizing the data in 100*100 


for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path 
    class_num = CATEGORIES.index(category)  # classification  0=dog 1=cat

    for img in tqdm(os.listdir(path)):  # iterating over each image
        try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
            training_data.append([new_array, class_num])  # adding the gray scaled and resized image to our training_data
        except Exception as e:  
            pass


print(len(training_data))


# In[ ]:


import random
random.shuffle(training_data)  


# In[ ]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))


# In[ ]:


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


#to save our preprocessed data we use pickel
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[ ]:


"""---------------Training the model-----------------"""


# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle


# In[2]:


#loading preprocessed data from pickels
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[3]:


X = X/255.0  #normalising the data-scaling the data


# In[4]:



model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[5]:

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)

# In[6]:

#using different combinations of number of dense layers , layer size and number of convolutional layer for determining the combination with maximum accuracy
import time
from tensorflow.keras.callbacks import TensorBoard


dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]


# In[13]:



for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

            model.fit(X, y,batch_size=32,epochs=7,validation_split=0.3,callbacks=[tensorboard])


# In[15]:



#combination of dense layer , layer size and convolutional layer that gives maximum accuracy
#log - 3-conv-64-nodes-0-dense-1602491963
#gave loss: 0.3156 - acc: 0.8619 - val_loss: 0.3739 - val_acc: 0.8389
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

            model.fit(X, y,batch_size=32,epochs=10,validation_split=0.2,callbacks=[tensorboard])

model.save('64x3-CNN.model')


# In[16]:


#above model gives accuracy of
# loss: 0.2057 - acc: 0.9144 - val_loss: 0.3775 - val_acc: 0.8405


# In[ ]:


"""-----------------Using the model for prediction-----------------"""


# In[56]:




CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value
import cv2
import os
import matplotlib.pyplot as plt

def prepare(path):
    
    IMG_SIZE =100
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    plt.imshow(img_array, cmap='gray')  # graph it
    plt.show()  

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[57]:


model = tf.keras.models.load_model("64x3-CNN.model")


# In[64]:


prediction = model.predict([prepare('dog1.jpg')]) 
print(CATEGORIES[int(prediction[0][0])])

