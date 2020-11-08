# Importing libraries
import os
import cv2
import numpy as np

features = []   # For storing image array 
labels = []     # For storing image label
classes = 2     # Two classes dog and cat

# Image pre-processing 
for i in range(classes):
    path = os.path.join('../TrainImages',str(i))
    images = os.listdir(path)
    for imgName in images:
        try:
            img_path = os.path.join(path,imgName)
            image = cv2.imread(img_path)        # Reading image
            image = cv2.resize(image,(50,50))   # Image resizing 
            image = np.array(image)             # Converting image array to numpy array
            features.append(image)              # Storing image array 
            labels.append(i)                    # Storing image label
        except Exception as e:
            pass
           
# Converting list to numpy array
features = np.array(features)
labels = np.array(labels)
print("Total features accessed : ", len(features))
print("Shape of image : ", features[0].shape)

# Data saving
np.save('features', features)
np.save('labels', labels)