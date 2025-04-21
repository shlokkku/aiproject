import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow_addons as tfa
import time

# Page configuration
st.set_page_config(
    page_title="Thalassemia Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #636EFA;
        margin-bottom: 2rem;
        text-align: center;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #9e9e9e;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_H5 = "aiprojectmodel.h5"
DRIVE_FILE_ID = "1NdV8NUwQAEtASsaV8F31dc5nKobbS0AI"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4914

# Sidebar
with st.sidebar:
    st.image("https://www.svgrepo.com/show/39225/blood.svg", width=100)
    st.markdown("## About")
    st.info("""
    This application uses deep learning to detect signs of Thalassemia 
    from blood smear images. Upload your microscopic blood smear image 
    to get an instant analysis.
    """)
    
    st.markdown("## What is Thalassemia?")
    st.markdown("""
    Thalassemia is an inherited blood disorder characterized by 
    reduced hemoglobin production. Early detection is crucial for 
    proper management and treatment.
    """)
    
    st.markdown("## Model Information")
    st.markdown("""
    - Architecture: Efficient net V2M
    - Input size: 224×224 pixels
    - Training accuracy: ~95%
    """)
    
    with st.expander("⚙️ Advanced Settings"):
        custom_threshold = st.slider(
            "Detection Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=THRESHOLD,
            step=0.05,
            help="Adjust the sensitivity of the model"
        )

# Main Content
st.markdown("<h1 class='main-header'>🩸 Thalassemia Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>AI-Powered Blood Smear Analysis</p>", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### Upload Blood Smear Image")
    st.markdown("Please upload a clear microscopic image of a blood smear.")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Download and load model (only once)
@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_H5):
        with st.spinner("📦 Downloading model... This may take a moment"):
            gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_H5, quiet=False)
    
    # Define custom objects dictionary
    custom_objects = {
        "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
    }
    
    # Load model with custom objects
    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(MODEL_H5, compile=False)

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Process image and make prediction
if uploaded_file is not None:
    with col2:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### Analysis Results")
        
        # Preprocess image
        image_processed = Image.open(uploaded_file).convert("RGB")
        image_processed = image_processed.resize(IMG_SIZE)
        img_array = np.array(image_processed) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Show processing animation
        with st.spinner("🔍 Analyzing blood smear..."):
            # Add a slight delay to show the spinner (better UX)
            time.sleep(1.5)
            
            try:
                # Make prediction
                prediction = model.predict(img_array)[0][0]
                threshold = custom_threshold if 'custom_threshold' in locals() else THRESHOLD
                label = "Thalassemia" if prediction > threshold else "Normal"
                
                # Display result
                result_color = "#FF4B4B" if label == "Thalassemia" else "#4CAF50"
                emoji = "🟥" if label == "Thalassemia" else "🟩"
                
                st.markdown(f"""
                <div class='result-box' style='background-color: {result_color}20;'>
                    <h2 style='color: {result_color};'>{emoji} {label}</h2>
                    
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown("#### Confidence Meter")
                st.progress(float(prediction))
                
                # Recommendation
                if label == "Thalassemia":
                    st.warning("""
                    **Recommendation:** The analysis suggests potential signs of Thalassemia. 
                    Please consult with a hematologist for proper diagnosis and assessment.
                    """)
                else:
                    st.success("""
                    **Recommendation:** The analysis suggests normal blood cell morphology.
                    For comprehensive health assessment, regular medical check-ups are advised.
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
**Disclaimer:** This tool is for educational and screening purposes only. 
It is not a substitute for professional medical diagnosis.
Always consult healthcare professionals for medical advice.
""")
st.markdown("</div>", unsafe_allow_html=True)
