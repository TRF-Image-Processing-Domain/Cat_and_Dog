import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ['Dog', 'Cat']

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (150, 150))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, 150, 150, 1)

model = tf.keras.models.load_model("DogsVCats.model")

prediction = model.predict([prepare('cat.jpg')])
print(CATEGORIES[int(prediction[0][0])])