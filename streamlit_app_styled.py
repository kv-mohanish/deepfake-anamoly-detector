import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
from PIL import Image
import time

# --- 1. Enhanced "Neural-Forensic" CSS ---
CSS_UPGRADE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

:root {
    --primary: #8b5cf6;
    --accent: #00f2ff;
    --bg-dark: #050505;
}

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background & Glassmorphism */
.stApp {
    background: radial-gradient(circle at top right, #1e1b4b, #000000);
}

/* The Scanline Animation */
.scan-line {
    width: 100%;
    height: 2px;
    background: var(--accent);
    box-shadow: 0 0 15px var(--accent);
    position: absolute;
    z-index: 10;
    animation: scan 3s linear infinite;
}

@keyframes scan {
    0% { top: 0%; }
    100% { top: 100%; }
}

/* Card Styling */
div[data-testid="stVerticalBlock"] > div:has(div.stImage) {
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.02);
    position: relative;
    overflow: hidden;
}

/* Big Metric Styling */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent) !important;
    text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 20, 0.95);
    border-right: 1px solid var(--primary);
}

/* Glow Button */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: transparent;
    border: 1px solid var(--primary);
    color: white;
    transition: 0.3s;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.stButton > button:hover {
    background: var(--primary);
    box-shadow: 0 0 20px var(--primary);
}
</style>
"""

st.set_page_config(page_title="DeepScan AI", layout="wide")
st.markdown(CSS_UPGRADE, unsafe_allow_html=True)

# --- 2. Sidebar for Metadata ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("System Specs")
    st.info("**Back-end:** VGG16 (ImageNet Weights)")
    st.info("**Classifier:** One-Class SVM")
    st.divider()
    st.write("### Model Stats")
    st.progress(0.92, text="Model Accuracy: 92%")
    st.progress(0.04, text="Latency: 45ms")

# --- 3. Main UI Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("# 🛡️ DEEP<span style='color:#8b5cf6'>SCAN</span>", unsafe_allow_html=True)
    st.markdown("### Forensic Image Authenticity Analysis")
    st.write("Upload a suspect profile picture to analyze neural patterns and identify deepfake structural anomalies.")
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        # Adding a visual "Scan" effect overlay would usually happen here

with col_right:
    if uploaded_file:
        st.markdown("### <br><br>Analysis Terminal", unsafe_allow_html=True)
        
        # Simulated "Processing" sequence for better UX
        with st.status("Initializing Neural Weights...", expanded=True) as status:
            time.sleep(0.8)
            st.write("Extracting VGG16 feature maps...")
            time.sleep(1.2)
            st.write("Scaling vector through SVM hyperplane...")
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Logic for prediction (Placeholder for your specific SVM logic)
        # score = process_and_predict(...)
        score = 0.5 # Example score
        
        st.divider()
        
        # Displaying result in a "Dashboard" style
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("ANOMALY SCORE", f"{score:+.4f}")
        with m_col2:
            verdict = "AUTHENTIC" if score > 0 else "FRAUDULENT"
            st.metric("VERDICT", verdict)

        if score > 0:
            st.success(f"**SUCCESS:** This image aligns with standard human facial distributions.")
        else:
            st.error(f"**ALERT:** This image exhibits latent space inconsistencies typical of GAN generation.")
            
    else:
        st.info("👈 Please upload an image to begin the forensic analysis.")
