import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import zipfile

# Constants
MODEL_ZIP = "aiprojectmodel.zip"
MODEL_DIR = "aiprojectmodel"
DRIVE_FILE_ID = "1AzFzmZkjasUFMjCC6DjGfAhbr6xP7FfB"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4914  # Change this if you found a better threshold

# Download and unzip the model if not already present
if not os.path.exists(MODEL_DIR):
    with st.spinner("📦 Downloading and extracting model..."):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP, quiet=False)

        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")

# Load the model
model = tf.keras.models.load_model(MODEL_DIR)

# Streamlit UI
st.title("🔬 Thalassemia Detection")
st.write("Upload a blood smear image to detect signs of Thalassemia.")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "Thalassemia" if prediction > THRESHOLD else "Normal"
    color = "🟥" if label == "Thalassemia" else "🟩"

    st.markdown(f"## {color} Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{prediction:.2f}`")
