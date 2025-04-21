import streamlit as st
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow_addons as tfa  # Make sure this import is present

# Constants
MODEL_H5 = "aiprojectmodel.h5"
DRIVE_FILE_ID = "1NdV8NUwQAEtASsaV8F31dc5nKobbS0AI"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4914  # or replace with your best threshold

# Download model if not present
if not os.path.exists(MODEL_H5):
    with st.spinner("📦 Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_H5, quiet=False)

# Define custom objects dictionary
custom_objects = {
    "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
}

# Load model with custom objects
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_H5, compile=False)

# Rest of your code remains the same
# ...

# App UI
st.title("🩸 Thalassemia Detection App")
st.write("Upload a blood smear image to detect signs of Thalassemia.")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Debugging: Check input shape
    st.write(f"Image Shape: {img_array.shape}")

    # Predict
    try:
        prediction = model.predict(img_array)[0][0]
        label = "Thalassemia" if prediction > THRESHOLD else "Normal"
        emoji = "🟥" if label == "Thalassemia" else "🟩"
        st.markdown(f"## {emoji} Prediction: **{label}**")
       
    except Exception as e:
        st.error(f"Error making prediction: {e}")
