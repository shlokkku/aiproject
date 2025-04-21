import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow_addons as tfa
import gc

# Set page config
st.set_page_config(
    page_title="Thalassemia Detection",
    page_icon="🔬",
    layout="centered"  # Changed to centered for less memory usage
)

# Simpler CSS with minimal styling
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #FF4B4B; text-align: center; }
    .subheader { font-size: 1.2rem; color: #636EFA; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_H5 = "aiprojectmodel.h5"
DRIVE_FILE_ID = "1NdV8NUwQAEtASsaV8F31dc5nKobbS0AI"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4914

# Main Content - Simpler layout
st.markdown("<h1 class='main-header'>🩸 Thalassemia Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload a blood smear image for analysis</p>", unsafe_allow_html=True)

# Add info in an expander to save space
with st.expander("About Thalassemia"):
    st.markdown("""
    Thalassemia is an inherited blood disorder characterized by 
    reduced hemoglobin production. This app analyzes blood smear images 
    to detect potential signs of the condition.
    """)

# Only load model when needed (lazy loading)
@st.cache_resource(show_spinner=False)
def get_model():
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
    return model

# Upload interface
uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg", "jpeg", "png"])

# Only process when there's an upload
if uploaded_file is not None:
    # Load image and display
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)  # Reduced width to save memory
    
    # Load model only when needed
    with st.spinner("Preparing analysis..."):
        try:
            model = get_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    # Process image
    with st.spinner("Analyzing image..."):
        try:
            # Preprocess - resize before converting to numpy to save memory
            image_small = image.resize(IMG_SIZE)
            img_array = np.array(image_small) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Free up original image memory
            del image
            del image_small
            gc.collect()
            
            # Predict
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            # Free prediction memory
            del img_array
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Display results
            label = "Thalassemia" if prediction > THRESHOLD else "Normal"
            emoji = "🟥" if label == "Thalassemia" else "🟩"
            
            st.markdown(f"### {emoji} Result: {label}")
           
            
            # Recommendation based on result
            if label == "Thalassemia":
                st.warning("Consider consulting with a healthcare professional for further evaluation.")
            else:
                st.success("Blood smear appears normal based on analysis.")
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            # Clean up on error
            tf.keras.backend.clear_session()
            gc.collect()

# Disclaimer
st.caption("""
**Disclaimer:** This tool is for educational purposes only and not a substitute for professional medical diagnosis.
""")
