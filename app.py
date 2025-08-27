# potato_disease_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_disease_model.h5")

model = load_model()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ------------------------------
# Preprocess Image
# ------------------------------
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🥔 Potato Plant Disease Prediction")
st.write("Upload a potato leaf image to predict its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("⏳ Processing...")
    img_array = preprocess_image(image)

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"✅ Predicted Class: **{predicted_class}**")
    st.info(f"🔍 Confidence: {confidence:.2f}%")
