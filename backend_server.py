import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import io

# --- 1. Initialize Flask App ---
app = Flask(__name__)
# Enable CORS to allow requests from your HTML file
CORS(app)

# --- 2. Load Pre-trained Models ONCE on startup ---
print("Loading VGG16 model...")
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
print("VGG16 model loaded.")

print("Loading saved SVM model and scaler...")
try:
    if_model = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("SVM model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: 'svm_model.joblib' or 'scaler.joblib' not found.")
    print("Please run 'train_model.py' first to create these files.")
    exit()

# --- 3. Define the Prediction Function ---
def process_and_predict(img_bytes):
    """
    Takes image bytes, processes them, and returns a prediction.
    """
    try:
        # Load image from bytes
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        
        # Pre-process for VGG16
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
        # Extract features
        feature_vector = vgg_model.predict(img_array, verbose=0).flatten()
        
        # Scale features
        scaled_feature = scaler.transform([feature_vector])
        
        # Get prediction score
        score = if_model.decision_function(scaled_feature)[0]
        
        return score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- 4. Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
        # Read image file as bytes
        img_bytes = file.read()
        
        # Get prediction
        score = process_and_predict(img_bytes)
        
        if score is not None:
            # --- MODIFIED RESPONSE ---
            # Only send back the score
            return jsonify({
                'score': f"{score:.4f}"
            })
        else:
            return jsonify({'error': 'Failed to process image'}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0', port=5000)