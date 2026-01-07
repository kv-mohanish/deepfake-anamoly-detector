import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
from PIL import Image

# --- 1. CSS Injection for Styling ---
# Includes the original Glassmorphism base + Responsive & Interactive improvements
CSS_TO_INJECT = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* --- GLOBAL STYLES --- */
body {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
    color: #e0e0e0;
}

* {
    font-family: 'Space Grotesk', sans-serif;
}

/* --- MAIN CONTAINER (Glass Effect) --- */
[data-testid="stAppViewContainer"] > .main {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 1.5rem;
    padding: 3rem 2rem;
    margin: 1rem;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.25);
}

/* --- TYPOGRAPHY --- */
h1 {
    font-size: 3.5rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #a78bfa, #f472b6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

p, div.stMarkdown, .stText {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #c4b5fd; /* purple-200 */
}

/* Image Captions */
.stImage > div > div > div {
    color: #a78bfa;
    text-align: center;
    font-size: 0.9rem;
    margin-top: 5px;
}

/* --- FILE UPLOADER --- */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167, 139, 250, 0.3);
    border-radius: 1rem;
    padding: 2rem;
    background: rgba(139, 92, 246, 0.05);
    transition: all 0.3s ease-in-out;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(167, 139, 250, 0.8);
    background: rgba(139, 92, 246, 0.1);
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
}

[data-testid="stFileUploader"] label {
    color: #ffffff;
    font-size: 1.1rem;
}

/* --- BUTTONS --- */
.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #7c3aed);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.125rem;
    padding: 0.75rem 2.5rem;
    border-radius: 9999px;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #6d28d9);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
    transform: scale(1.02);
}
.stButton > button:focus {
    outline: 2px solid #a78bfa;
    outline-offset: 3px;
}

/* --- METRICS & ALERTS --- */
[data-testid="stMetric"] {
    text-align: center;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 1rem;
    padding: 1rem;
}
[data-testid="stMetric"] label {
    color: #c4b5fd; 
}
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 3.5rem;
    font-weight: 700;
    color: #fff;
}

/* Success/Error/Info Boxes */
[data-testid="stSuccess"], [data-testid="stError"], [data-testid="stInfo"], [data-testid="stWarning"] {
    border-radius: 0.5rem;
    backdrop-filter: blur(5px);
}
[data-testid="stSuccess"] {
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    color: #6ee7b7;
}
[data-testid="stError"] {
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #fca5a5;
}
[data-testid="stInfo"] {
    background-color: rgba(100, 116, 139, 0.2);
    border: 1px solid #64748b;
    color: #cbd5e1;
}

/* --- RESPONSIVENESS (MOBILE) --- */
@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] > .main {
        padding: 1.5rem 1rem;
        margin: 0.5rem;
    }
    h1 { font-size: 2.25rem; }
    
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
    
    .stButton > button {
        width: 100%; /* Full width buttons on mobile */
    }
}
"""

st.markdown(f"<style>{CSS_TO_INJECT}</style>", unsafe_allow_html=True)

# --- 2. Load Models ---
@st.cache_resource
def load_models():
    """
    Loads the VGG16 model for feature extraction and the pre-trained
    Isolation Forest (or SVM) model and scaler.
    """
    st.info("Loading AI models... This may take a moment.", icon="⏳")
    try:
        # Load VGG16 model without the top classification layer
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        # Load the pre-trained anomaly detection model and scaler
        # Ensure these files exist in your directory
        if_model = joblib.load('svm_model.joblib') 
        scaler = joblib.load('scaler.joblib')
        
        # Clear the info box by replacing it with success (optional, or just pass)
        st.success("System ready. Models loaded.", icon="✅")
        return vgg_model, if_model, scaler
    except FileNotFoundError:
        st.error(
            "**Model files missing.**\nPlease ensure `svm_model.joblib` and `scaler.joblib` "
            "are in the project directory.",
            icon="❌"
        )
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}", icon="❌")
        return None, None, None

vgg_model, if_model, scaler = load_models()

# --- 3. Prediction Function ---
def process_and_predict(img_bytes, vgg_model, if_model, scaler):
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
        st.error(f"Prediction Error: {e}", icon="❌")
        return None

# --- 4. Build the Streamlit UI ---
st.title("Deepfake Guard") 
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2.5rem;'>
        <p>Upload an image to analyze visual artifacts and detect potential anomalies using VGG16 + SVM.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if vgg_model is not None and if_model is not None:
    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()

        # Layout: Image on Left, Results on Right
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(img_bytes, caption="Input Image", use_column_width=True)

        with col2:
            st.write("### Analysis Results")
            with st.spinner("Extracting features..."):
                score = process_and_predict(img_bytes, vgg_model, if_model, scaler)

            if score is not None:
                st.metric(label="Anomaly Score", value=f"{score:.4f}")
                
                # Threshold logic (Adjust -10 based on your specific training data)
                THRESHOLD = -10 
                
                if score < THRESHOLD:
                    st.error("**Suspected Deepfake**", icon="🚨")
                    st.markdown(
                        """
                        The image features deviate significantly from real samples.
                        * **High anomaly score detected.**
                        * *Check for visual artifacts around eyes/mouth.*
                        """
                    )
                else:
                    st.success("**Likely Authentic**", icon="shield")
                    st.markdown(
                        """
                        The image features align with the training distribution of real images.
                        * **Low anomaly score detected.**
                        """
                    )
else:
    # Fallback if models failed to load
    st.warning("Application halted due to missing models.", icon="⚠️")
