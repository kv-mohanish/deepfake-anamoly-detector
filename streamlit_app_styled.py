import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
from PIL import Image

# --- 1. CSS Injection for Styling ---
# This section applies custom CSS to style the Streamlit application,
# aiming for a modern, "glass-card" aesthetic with a specific font.
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

# --- 2. Load Models ---
# @st.cache_resource decorator ensures that the models are loaded only once
# across all reruns of the Streamlit app, improving performance.
@st.cache_resource
def load_models():
    """
    Loads the VGG16 model for feature extraction and the pre-trained
    Isolation Forest (or SVM) model and scaler for anomaly detection.
    """
    st.info("Loading AI models... This may take a moment.", icon="⏳")
    try:
        # Load VGG16 model without the top classification layer,
        # using ImageNet weights, and applying average pooling.
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        st.success("VGG16 feature extractor loaded successfully.", icon="✅")

        # Load the pre-trained anomaly detection model (e.g., Isolation Forest)
        # and the scaler used during its training.
        if_model = joblib.load('svm_model.joblib') # Assuming 'svm_model.joblib' contains the Isolation Forest model
        scaler = joblib.load('scaler.joblib')
        st.success("Anomaly detection model and scaler loaded successfully.", icon="✅")
        return vgg_model, if_model, scaler
    except FileNotFoundError:
        st.error(
            "Error: Model files not found. Please ensure 'svm_model.joblib' and 'scaler.joblib' "
            "are in the same directory as this script. Run 'train_model.py' first if you haven't.",
            icon="❌"
        )
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}", icon="❌")
        return None, None, None

# Attempt to load models at the start of the application.
vgg_model, if_model, scaler = load_models()

# --- 3. Prediction Function ---
def process_and_predict(img_bytes, vgg_model, if_model, scaler):
    """
    Processes an uploaded image, extracts features using VGG16,
    and predicts an anomaly score using the loaded model.

    Args:
        img_bytes (bytes): The raw bytes of the uploaded image.
        vgg_model (tf.keras.Model): The pre-loaded VGG16 model.
        if_model: The pre-loaded anomaly detection model (e.g., Isolation Forest).
        scaler: The pre-loaded scaler.

    Returns:
        float: The anomaly score, or None if an error occurred.
    """
    try:
        # Open image from bytes, convert to RGB (to handle potential RGBA images), and resize.
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224)) # VGG16 expects 224x224 input.

        # Convert PIL image to NumPy array and add batch dimension.
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for VGG16 (e.g., mean subtraction, scaling).
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        # Extract features using VGG16 and flatten the output.
        # verbose=0 suppresses prediction progress bar.
        feature_vector = vgg_model.predict(img_array, verbose=0).flatten()

        # Scale the feature vector using the pre-trained scaler.
        scaled_feature = scaler.transform([feature_vector])

        # Get the anomaly score from the model.
        # Isolation Forest's decision_function returns negative values for anomalies.
        score = if_model.decision_function(scaled_feature)[0]
        return score
    except Image.UnidentifiedImageError:
        st.error("The uploaded file is not a valid image format.", icon="⚠️")
        return None
    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}", icon="❌")
        return None

# --- 4. Build the Streamlit UI ---
st.title("AI-Powered Anomaly Detector")
st.markdown(
    "<p style='text-align: center; font-size: 1.25rem; color: #c4b5fd; margin-top: -1rem; margin-bottom: 2rem;'>"
    "Analyze images for authenticity and detect potential anomalies."
    "</p>",
    unsafe_allow_html=True
)

# Only proceed if models were loaded successfully.
if vgg_model is None or if_model is None or scaler is None:
    st.warning("Please ensure all model files are correctly placed and try again.", icon="⚠️")
else:
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG) to analyze",
        type=["jpg", "png", "jpeg"],
        help="The detector analyzes visual features to identify images that deviate significantly from a 'normal' dataset."
    )

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()

        # Use columns to display the image and results side-by-side.
        col1, col2 = st.columns([0.6, 0.4]) # Image takes 60% width, results take 40%.

        with col1:
            st.image(img_bytes, caption="Uploaded Image", use_column_width=True, output_format='auto')

        with col2:
            # Display a spinner while analysis is in progress.
            with st.spinner("Analyzing image for anomalies..."):
                score = process_and_predict(img_bytes, vgg_model, if_model, scaler)

            if score is not None:
                st.metric(label="Anomaly Score", value=f"{score:+.4f}")

                # Interpret the anomaly score. Lower (more negative) scores indicate higher anomaly likelihood.
                # The threshold (-10) is an example and might need tuning based on your model's training.
                if score < -10:
                    st.error(f"**Result: Likely Anomalous / Deepfake**", icon="🚨")
                    st.write(
                        "This image exhibits features statistically different from the dataset of 'normal' images "
                        "it was trained on. This could indicate manipulation or an unusual pattern."
                    )
                else:
                    st.success(f"**Result: Likely Authentic**", icon="👍")
                    st.write(
                        "This image's features are statistically similar to the dataset of 'normal' images "
                        "it was trained on, suggesting it is likely authentic."
                    )
            else:
                # Error message already handled within process_and_predict, but this catches any remaining None.
                st.error("Failed to analyze the image. Please try another file.", icon="❌")

    else:
        st.info("Upload an image above to begin the anomaly detection process.", icon="⬆️")

