# Import necessary libraries
import streamlit as st  # For creating the web interface
from keras.models import load_model  # For loading the trained neural network
from PIL import Image, ImageOps  # For image processing
import numpy as np  # For numerical operations
import tensorflow as tf  # Deep learning framework

def classify(image, model, class_names):
    """
    Main classification function that processes an image and returns predictions
    
    Args:
        image: Input image to classify
        model: Loaded Keras model
        class_names: List of class names (tick bite vs mosquito bite)
    
    Returns:
        tuple: (predicted class name, confidence score)
    """
    # Resize image to 224x224 pixels (standard input size for many CNN models)
    # LANCZOS is a high-quality resampling method that preserves image quality
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert PIL Image to numpy array for mathematical operations
    image_array = np.asarray(image)

    # Normalize pixel values to range [-1, 1] instead of [0, 255]
    # This is a common preprocessing step for neural networks
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create input batch of size 1 (single image)
    # Shape (1, 224, 224, 3) represents:
    # 1: batch size
    # 224, 224: image dimensions
    # 3: RGB channels
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Get model's prediction probabilities for each class
    prediction = model.predict(data)

    # Get index of highest probability class
    index = np.argmax(prediction)

    # Get corresponding class name and confidence score
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score

# Set up the Streamlit web interface
st.title('Bug Bite Classification')
st.header('Please upload an image of the bite')

# Create file uploader widget
# Accepts only image files (jpeg, jpg, png)
# label_visibility="collapsed" hides the default "Choose an image..." text
file = st.file_uploader("Choose an image...", type=['jpeg', 'jpg', 'png'], 
                       label_visibility="collapsed")

# Load the pre-trained model from disk
model = load_model('keras_model.h5')

# Read class names from labels file
# Format expected: "0 tick_bite" or "1 mosquito_bite" (index space label)
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]

# Process uploaded image if one exists and model is loaded
if file is not None and model:
    # Open image and convert to RGB (handles PNG transparency)
    image = Image.open(file).convert('RGB')
    
    # Display the uploaded image
    st.image(image, use_column_width=True)

    # Get prediction
    class_name, conf_score = classify(image, model, class_names)

    # Display results
    st.write(f"## {class_name}")  # Show predicted class
    st.write(f"### Score: {conf_score * 100:.1f}%")  # Show confidence as percentage