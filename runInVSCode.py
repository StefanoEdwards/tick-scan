# Imports necessary libraries
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Disables scientific notation for clarity
np.set_printoptions(suppress=True)

# Loads the pre-trained neural network model
model = load_model("keras_model.h5", compile=False)

# Loads the class labels from a file
class_names = open("labels.txt", "r").readlines()
category_to_index = {}
for class_name in class_names:
    index, name = class_name.split(' ')
    name = name[:-1]
    category_to_index[class_name] = int(index)

image_path = "C:\\Users\\SEdwards\\Downloads\\CV Data\\data\\validation\\tick\\2020-09-13_12-25-15.jpg"

# Creates an empty numpy array to store the image data
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Opens the image file and converts it to RGB format
image = Image.open(image_path).convert("RGB")

# Resizes and crop the image to fit the model's input size
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Converts the image to a numpy array
image_array = np.asarray(image)

# Normalizes the pixel values of the image array to be in the range [-1, 1]
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Loads the normalized image data into the numpy array
data[0] = normalized_image_array

# Passes the image data through the pre-trained model to get predictions
prediction = model.predict(data)

# Finds the index of the class with the highest probability prediction
index = np.argmax(prediction)

# Retrieves the predicted class name from the class labels
class_name = class_names[index]

# Gets the confidence score (probability) corresponding to the predicted class
confidence_score = prediction[0][index]

# Prints the predicted class and its confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)