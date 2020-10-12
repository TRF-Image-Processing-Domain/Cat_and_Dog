# Importing libraries
import os
import cv2 
import numpy as np
from keras.models import load_model

# Loading trained model
model = load_model("../TrainedModels/L=46-A=79.model")

# Two classifications 
classes = { 0:'Dog',
            1:'Cat',  }

# Test image directory
imgPath = os.listdir('../TestImages')

# Testing our trained model with test images
for img in imgPath:
    testFeature = []
    path = os.path.join("../TestImages", img)   # Setting path
    image = cv2.imread(path)                    # Reading test image
    resized = cv2.resize(image,(50,50))         # Resizing the input image
    testFeature.append(np.array(resized))       # Storing image array
    f_test = np.array(testFeature)              # Converting to numpy array for prediction
    f_pred = model.predict_classes(f_test)      # Predicting results
    print("Predicted animal : ", classes[f_pred[0][0]], "\tExpected animal : ", img)
    cv2.putText(image, "Prediction: " + str(classes[f_pred[0][0]]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow(img, image)
    cv2.waitKey(0)
cv2.destroyAllWindows()