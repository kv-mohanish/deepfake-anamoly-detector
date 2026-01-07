import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
from PIL import Image
import pymongo  # New dependency
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Deepfake Anomaly Detector",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CYBERPUNK CSS (Truncated for brevity, keep your original CSS)
# =========================================================
CSS = """
/* Your existing CSS remains here... */
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# =========================================================
# DATABASE & MODEL INITIALIZATION
# =========================================================
@st.cache_resource
def init_systems():
    # 1. Load ML Models
    vgg = VGG16(weights="imagenet", include_top=False, pooling="avg")
    svm = joblib.load("svm_model.joblib")
    scaler = joblib.load("scaler.joblib")
    
    # 2. Connect to MongoDB Atlas
    # Replace with your actual password if different
    uri = "mongodb+srv://kvmohanish_db_user:deepfakepassword123@cluster0.tteyk9i.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(uri)
    db = client["DeepfakeDetection"]
    collection = db["analysis_logs"]
    
    return vgg, svm, scaler, collection

try:
    vgg_model, svm_model, scaler, logs_col = init_systems()
    system_ok = True
except Exception as e:
    st.error(f"SYSTEM CRITICAL ERROR: {e}")
    system_ok = False

# =========================================================
# SIDEBAR TASKBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚡ CONTROL NODE")
    st.success("NEURAL CORE & ATLAS ONLINE" if system_ok else "SYSTEM OFFLINE")

    st.markdown("---")
    st.markdown("### 📊 CLOUD TELEMETRY (Recent Scans)")
    if system_ok:
        # Fetch last 5 logs from Atlas
        recent_logs = logs_col.find().sort("timestamp", -1).limit(5)
        for log in recent_logs:
            status = "🛑 FAIL" if log.get("is_anomaly") else "🛡️ PASS"
            st.code(f"{log['timestamp'].strftime('%H:%M:%S')} | {status} | Score: {log['score']:.2f}", language="text")

    st.markdown("---")
    THRESHOLD = st.slider("ANOMALY THRESHOLD", -50.0, 0.0, -10.0, 0.5)

# =========================================================
# PREDICTION & STORAGE FUNCTION
# =========================================================
def analyze_and_log(img_bytes, filename):
    # Image Preprocessing
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    
    # Feature Extraction & SVM Inference
    features = vgg_model.predict(arr, verbose=0).flatten()
    features = scaler.transform([features])
    score = float(svm_model.decision_function(features)[0]) # Convert to float for Mongo
    
    # Save to MongoDB Atlas
    if system_ok:
        log_entry = {
            "timestamp": datetime.now(),
            "filename": filename,
            "score": score,
            "threshold": THRESHOLD,
            "is_anomaly": bool(score < THRESHOLD),
            "dev_crew": ["CHIRAG", "MOHANISH", "PRANEETH", "THRISHAL"]
        }
        logs_col.insert_one(log_entry)
        
    return score

# =========================================================
# MAIN UI
# =========================================================
st.markdown("""<div style="text-align:center;"> <h1 class="glitch">DEEPFAKE ANOMALY DETECTOR</h1> </div>""", unsafe_allow_html=True)
st.markdown('<div class="pipeline"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("UPLOAD IMAGE FOR ANALYSIS", type=["jpg", "jpeg", "png"])

if uploaded_file and system_ok:
    img_bytes = uploaded_file.read()
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.markdown('<div class="card-cyan">#### 🔍 INPUT NODE</div>', unsafe_allow_html=True)
        st.image(img_bytes, use_column_width=True)

    with col2:
        st.markdown('<div class="card-pink scanline">#### 🧠 ANALYSIS CORE</div>', unsafe_allow_html=True)
        with st.spinner("⚡ SYNCING WITH ATLAS & ANALYZING..."):
            score = analyze_and_log(img_bytes, uploaded_file.name)

        st.metric("🧠 ANOMALY SCORE", f"{score:.4f}")
        
        if score < THRESHOLD:
            st.error("🛑 SYNTHETIC PATTERN DETECTED")
        else:
            st.success("🛡️ STRUCTURE WITHIN NORMAL BOUNDARY")
        st.markdown("</div>", unsafe_allow_html=True)
