import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_PATH = "final_thalassemia_model.h5"
DRIVE_FILE_ID = "1TAtCgpnA3HEnQBPC-asLh0-5OTyJ0H9E"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # Or use best_threshold if known

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🔬 Thalassemia Detection App")
st.write("Upload a blood smear image to detect Thalassemia presence.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "Thalassemia" if prediction > THRESHOLD else "Normal"
    color = "🟥" if label == "Thalassemia" else "🟩"

    st.markdown(f"## {color} Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{prediction:.2f}`")
