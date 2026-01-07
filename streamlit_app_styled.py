import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import joblib
import io
import pymongo 
from datetime import datetime 
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Deepfake Anomaly Detector",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CYBERPUNK CSS
# =========================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;500&display=swap');

body {
    background-color: #020207;
    color: #e0e0e0;
}

* {
    font-family: 'Roboto Mono', monospace;
}

h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 2px;
    text-transform: uppercase;
}

[data-testid="stAppViewContainer"] > .main {
    background:
        linear-gradient(rgba(0,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.03) 1px, transparent 1px),
        radial-gradient(circle at 20% 30%, rgba(188,19,254,0.08), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(0,243,255,0.08), transparent 40%);
    background-size: 30px 30px, 30px 30px, cover, cover;
    border: 1px solid #00f3ff;
    box-shadow: 0 0 30px rgba(0,243,255,0.15);
    padding: 2.5rem;
    max-width: 96%;
    margin: 1.5rem auto;
}

/* GLITCH TITLE */
.glitch {
    color: #00f3ff;
    text-shadow: 2px 0 #bc13fe, -2px 0 #00ffea;
    
}
@keyframes glitch {
    0% { transform: translate(0); }
    20% { transform: translate(-2px, 2px); }
    40% { transform: translate(2px, -2px); }
    60% { transform: translate(-1px, 1px); }
    80% { transform: translate(1px, -1px); }
    100% { transform: translate(0); }
}

/* PIPELINE */
.pipeline {
    height: 6px;
    background: linear-gradient(90deg, transparent, #00f3ff, transparent);
    position: relative;
    overflow: hidden;
    margin: 20px 0;
}
.pipeline::after {
    content: "";
    position: absolute;
    width: 120px;
    height: 100%;
    left: -150px;
    background: linear-gradient(90deg, transparent, rgba(188,19,254,0.8), transparent);
    animation: flow 3s linear infinite;
}
@keyframes flow {
    from { left: -150px; }
    to { left: 100%; }
}

/* CARDS */
.card-cyan {
    border: 1px solid #00f3ff;
    padding: 1.2rem;
    box-shadow: 0 0 20px rgba(0,243,255,0.25);
}
.card-pink {
    border: 1px solid #bc13fe;
    padding: 1.2rem;
    box-shadow: 0 0 20px rgba(188,19,254,0.25);
}

/* NODE */
.node {
    width: 12px;
    height: 12px;
    background: #00f3ff;
    border-radius: 50%;
    box-shadow: 0 0 12px #00f3ff;
    animation: pulse 2s infinite alternate;
}
@keyframes pulse {
    from { transform: scale(1); opacity: 0.6; }
    to { transform: scale(1.6); opacity: 1; }
}

/* SCANLINE */
.scanline {
    position: relative;
    overflow: hidden;
}
.scanline::after {
    content: "";
    position: absolute;
    top: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(to bottom, transparent, rgba(0,243,255,0.15), transparent);
    animation: scan 3s linear infinite;
}
@keyframes scan {
    0% { top: -100%; }
    100% { top: 100%; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    vgg = VGG16(weights="imagenet", include_top=False, pooling="avg")
    svm = joblib.load("svm_model.joblib")
    scaler = joblib.load("scaler.joblib")
    
    # Database connection
    uri = "mongodb+srv://kvmohanish_db_user:deepfakepassword123@cluster0.tteyk9i.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(uri)
    db = client["DeepfakeDetection"]
    collection = db["analysis_logs"]
    
    return vgg, svm, scaler, collection

# Initialize global variables
vgg_model, svm_model, scaler, logs_col = None, None, None, None
system_ok = False

try:
    # Corrected unpacking: now expects 4 values
    vgg_model, svm_model, scaler, logs_col = load_models()
    system_ok = True
except Exception as e:
    st.error(f"System Load Error: {e}")
    system_ok = False
# =========================================================
# SIDEBAR TASKBAR (WITH TECH STACK)
# =========================================================
with st.sidebar:
    st.markdown("## ⚡ CONTROL NODE")
    st.success("NEURAL CORE ONLINE" if system_ok else "SYSTEM OFFLINE")

    st.markdown("---")
    st.markdown("### 🧩 TECH STACK")
    st.markdown("""
    ```text
    Frontend   : Streamlit
    Backend    : Python 3.x
    DL Engine  : TensorFlow / Keras
    ML Model   : One-Class SVM
    Vision     : VGG16 (ImageNet)
    Scaling    : StandardScaler
    Deployment : Local / Cloud
    ```
    """)
    st.markdown("---")
    st.markdown("### 🧠 MODEL CONFIG")
    st.markdown("""
    ```text            
    Learning Type : Unsupervised
    Input Size    : 224 x 224 RGB
    Feature Dim   : 512
    Decision Rule : Anomaly Score
    Dataset       : Authentic Faces
    ```            
    """)
    st.markdown("---")
    THRESHOLD = st.slider(
    "ANOMALY THRESHOLD",
    min_value=-50.0,
    max_value=0.0,
    value=-10.0,
    step=0.5
)

    st.markdown("---")
    st.markdown("""
    👾 DEV CREW
    ```text
    CHIRAG K
    MOHANISH K V 
    PRANEETH P K 
    THRISHAL     
     
    ```
                """)
    st.markdown("---")
    st.markdown("### 📊 CLOUD TELEMETRY")
    if system_ok:
        try:
            recent_logs = logs_col.find().sort("timestamp", -1).limit(5)
            for log in recent_logs:
                status = "🛑 FAIL" if log.get("is_anomaly") else "🛡️ PASS"
                ts = log['timestamp'].strftime('%H:%M:%S')
                st.code(f"{ts} | {status} | Score: {log['score']:.2f}", language="text")
        except Exception:
            st.code("Waiting for telemetry data...", language="text")
# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""

<div style="text-align:center;"> <h1 class="glitch">DEEPFAKE ANOMALY DETECTOR</h1> <h3 style="color:#bc13fe;">// ANOMALY SCORER FOR PROBABLE DEEPFAKES //</h3> </div> """, unsafe_allow_html=True)

st.markdown('<div class="pipeline"></div>', unsafe_allow_html=True)

st.markdown("""

<pre style="color:#00f3ff; background:rgba(0,0,0,0.6); padding:15px; border-left:3px solid #bc13fe; font-size:0.75rem;"> [BOOT] Initializing visual cortex... [LOAD] VGG16 weights synced [SYNC] Feature grid aligned [INFO] Awaiting image input... </pre>

""", unsafe_allow_html=True)
# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
"UPLOAD IMAGE FOR ANALYSIS",
type=["jpg", "jpeg", "png"]
)
# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict(img_bytes, filename):
    # Image Preprocessing
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    
    # Feature Extraction & SVM Inference
    features = vgg_model.predict(arr, verbose=0).flatten()
    features = scaler.transform([features])
    score = float(svm_model.decision_function(features)[0]) 
    
    # Save to MongoDB Atlas
    if system_ok:
        log_entry = {
            "timestamp": datetime.now(),
            "filename": filename,
            "score": score,
            "is_anomaly": bool(score < THRESHOLD)
        }
        logs_col.insert_one(log_entry)
        
    return score
# =========================================================
# MAIN UI
# =========================================================
if uploaded_file and system_ok:
    img_bytes = uploaded_file.read()
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown('<div class="card-cyan">', unsafe_allow_html=True)
        st.markdown("#### 🔍 INPUT NODE")
        st.image(img_bytes, use_column_width=True)
        st.markdown('<div class="node"></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-pink scanline">', unsafe_allow_html=True)
        st.markdown("#### 🧠 ANALYSIS CORE")

        with st.spinner("⚡ PROPAGATING THROUGH NEURAL PIPELINE..."):
            score = predict(img_bytes)

        st.metric("🧠 ANOMALY SCORE", f"{score:.4f}")

        confidence = max(min((score - THRESHOLD) / abs(THRESHOLD), 1), -1)
        st.progress((confidence + 1) / 2)

        if score < THRESHOLD:
            st.error("🛑 SYNTHETIC PATTERN DETECTED")
        else:
            st.success("🛡️ STRUCTURE WITHIN NORMAL BOUNDARY")

        st.markdown("</div>", unsafe_allow_html=True)
elif not system_ok:
    st.error("⚠️ SYSTEM OFFLINE. FAILED TO LOAD MODELS.")


