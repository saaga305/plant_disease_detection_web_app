import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
MODEL_PATH = 'pepper-model.h5'  # Replace with the actual path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = [ 'Bacterial Spot', 'Healthy']

# Set the title and description of your app
st.title("Pepper Disease Classification")
st.write("Upload an image of a pepper leaf to classify its disease.")

# Upload an image for prediction
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference with the model
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (128, 128))
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

# You can add more content to your Streamlit app as needed
# For example, you can include explanations, additional features, or visuals.
