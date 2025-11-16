<p align="center">🧠 AI-Powered Image Anomaly Detector</p>
<p align="center">Unsupervised Deepfake & Image Anomaly Detection</p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue" /> <img src="https://img.shields.io/badge/Framework-Flask-red" /> <img src="https://img.shields.io/badge/Model-One--Class%20SVM-purple" /> <img src="https://img.shields.io/badge/UI-Glassmorphism-0ABAB5" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p> <p align="center"> 🔍 Detect anomalies. 🛡️ Identify deepfakes. 🚀 All using unsupervised ML + deep visual features. </p>

<p align="center"> <img src="assets/demo.gif" width="600"> </p>
✨ Features

🎨 Glassmorphism UI with smooth animations & score ring

⚡ Real-Time Image Analysis with instant scoring

🧠 Unsupervised Deepfake Detection

🔥 VGG16 Feature Extraction (512-D embeddings)

🌐 Flask API Backend

🎛️ Optional Streamlit App

📦 Production-ready file structure

🧩 How It Works
User Upload → Frontend → Flask API
             → VGG16 Feature Extractor
             → StandardScaler
             → One-Class SVM
             → JSON Response {"score": -0.32}

💡 Score Meaning
Score	Interpretation
Positive	Image is similar to training data (Likely Real)
Negative	Statistical outlier (Likely Fake / Anomaly)
🚀 Getting Started
1️⃣ Install Dependencies
git clone <repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate

pip install -r requirements.txt

🏋️‍♂️ 2️⃣ Train the Model (One-Time Only)
python train_model.py


This generates:

svm_model.joblib
scaler.joblib

🖥️ 3️⃣ Run the Backend Server
python backend_server.py


Runs at:

http://127.0.0.1:5000

🌐 4️⃣ Launch the Frontend

Simply open:

index.html


Do not use VS Code "Preview".
Open it directly in Chrome / Firefox.

🎨 Alternative UI (Streamlit)
streamlit run streamlit_app_styled.py

📁 Project Structure
/
├── svm_model.joblib
├── scaler.joblib
│
├── train_model.py
├── backend_server.py
├── index.html
│
├── streamlit_app_styled.py
├── requirements.txt
└── README.md

📦 Tech Stack

Python 3.8+

TensorFlow / Keras (VGG16)

Scikit-learn (One-Class SVM, StandardScaler)

Flask

Streamlit (optional)

HTML + TailwindCSS

🛠️ API Endpoint
POST /predict
curl -X POST http://127.0.0.1:5000/predict -F "image=@sample.jpg"

Example Response
{
  "score": -0.2438
}

🔮 Future Enhancements

 Add ROC curve and evaluation metrics

 Add mobile-friendly UI

 Add GPU inference support

 Add ONNX export

📜 License

This project is licensed under the MIT License.

🌟 Support

If you like this project:

⭐ Star this repo
🐛 Report issues
📣 Share with others

