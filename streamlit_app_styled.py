import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
from PIL import Image

# --- 1. CSS Injection for Styling ---
# This is how we try to match your HTML file's "vibe"
CSS_TO_INJECT = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Main page background */
body {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
    color: #e0e0e0;
}

/* Set the font for everything */
* {
    font-family: 'Space Grotesk', sans-serif;
}

/* Main app container - apply glass-card effect */
[data-testid="stAppViewContainer"] > .main {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 1.5rem; /* 24px */
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.4);
}

/* Header text */
h1 {
    font-size: 3.75rem; /* 6xl */
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #a78bfa, #f472b6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

/* Style the file uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167, 139, 250, 0.3);
    border-radius: 1rem;
    padding: 1rem;
    background: rgba(139, 92, 246, 0.05);
}
[data-testid="stFileUploader"] > label {
    color: #ffffff;
    font-size: 1.125rem;
    font-weight: 500;
}
[data-testid="stFileUploader"] p {
    color: #c4b5fd; /* purple-200 */
}

/* Style the button */
.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #7c3aed);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.125rem; /* lg */
    padding: 0.75rem 2rem;
    border-radius: 9999px; /* full */
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #6d28d9);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    transform: scale(1.03);
}
.stButton > button:disabled {
    background: #4b5563; /* gray-600 */
    color: #9ca3af; /* gray-400 */
}

/* Style the metric (score) */
[data-testid="stMetric"] {
    text-align: center;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 1rem;
    padding: 1.5rem;
}
[data-testid="stMetric"] > label {
    font-size: 1.25rem; /* xl */
    color: #c4b5fd;
    font-weight: 500;
}
[data-testid="stMetric"] > div {
    font-size: 4rem; /* ~6xl */
    font-weight: 700;
}

/* Style the result text (success/error boxes) */
[data-testid="stSuccess"] {
    background-color: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    border-radius: 0.5rem;
    color: #6ee7b7;
}
[data-testid="stError"] {
    background-color: rgba(239, 68, 68, 0.2);
    border: 1px solid #ef4444;
    border-radius: 0.5rem;
    color: #fca5a5;
}
"""

st.markdown(f"<style>{CSS_TO_INJECT}</style>", unsafe_allow_html=True)

# --- 2. Load Models (same as before) ---
@st.cache_resource
def load_models():
    # ... (same model loading code as before) ...
    print("Loading VGG16 model...")
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    print("VGG16 model loaded.")

    print("Loading saved SVM model and scaler...")
    try:
        if_model = joblib.load('svm_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print("SVM model and scaler loaded successfully.")
        return vgg_model, if_model, scaler
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'train_model.py' first.")
        return None, None, None

vgg_model, if_model, scaler = load_models()

# --- 3. Prediction Function (same as before) ---
def process_and_predict(img_bytes, vgg_model, if_model, scaler):
    # ... (same prediction logic as before) ...
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        feature_vector = vgg_model.predict(img_array, verbose=0).flatten()
        scaled_feature = scaler.transform([feature_vector])
        score = if_model.decision_function(scaled_feature)[0]
        return score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- 4. Build the Streamlit UI ---
st.title("Anomaly Detector")
st.markdown("<p style='text-align: center; font-size: 1.25rem; color: #c4b5fd; margin-top: -1rem; margin-bottom: 2rem;'>AI-Powered Image Authenticity Analysis</p>", unsafe_allow_html=True)


if vgg_model is None or if_model is None:
    st.error("Model files not found. Please make sure `svm_model.joblib` and `scaler.joblib` are in the same directory.")
else:
    uploaded_file = st.file_uploader("Drop your image here or click to browse", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        
        col1, col2 = st.columns([0.6, 0.4]) # Give image 60% width, result 40%
        
        with col1:
            st.image(img_bytes, caption="Uploaded Image", use_column_width=True, output_format='auto')

        with col2:
            with st.spinner("Analyzing..."):
                score = process_and_predict(img_bytes, vgg_model, if_model, scaler)
            
            if score is not None:
                st.metric(label="Anomaly Score", value=f"{score:+.4f}")
                
                if score < 0:
                    st.error(f"**Result: Likely Deepfake / Anomaly**")
                    st.write("This image is statistically different from the training data of real faces.")
                else:
                    st.success(f"**Result: Likely Real**")
                    st.write("This image is statistically similar to the training data of real faces.")
            else:
                st.error("Could not process the image.")