# Importing libraries
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Loading 'features' and 'labels' file
features = np.load('../LoadData/features.npy')
labels = np.load('../LoadData/labels.npy')

# Data pre-processing
f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.1, random_state=0)

# Normalizing image array values
f_train = f_train/255.0
f_test = f_test/255.0

# CNN sequential model
model = Sequential()
# ConvLayer 1
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu', input_shape= (50, 50, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
# ConvLayer 2
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# DenseLayer 1 (hidden layer)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# Dropout for Overfitting purpose
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(f_train, l_train, batch_size=32, epochs=10, validation_data=(f_test, l_test))

# Calculate loss and accuracy
val_loss,val_acc=model.evaluate(f_test,l_test)
# Saving our trained model
model.save("../TrainedModels/L={}-A={}.model".format(int(val_loss*100), int(val_acc*100)))


# Predicting first 10 images from test dataset
pred = model.predict_classes(f_test)
a = 0
for i in range(10):
    if pred[i] == 1:
        a = a + 1
    print("Predicted class : ", pred[i][0], "\tExpected class : ", l_test[i])