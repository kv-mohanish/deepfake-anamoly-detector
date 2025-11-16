import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import tensorflow_datasets as tfds
import os
import joblib  # Import joblib to save the models

# --- Step 1: Data Preparation and Feature Extraction ---
def extract_features(dataset, model, max_images=2000): # Still 2000 images
    """
    Extracts features from a TensorFlow dataset using a pre-trained CNN.
    """
    features = []
    print(f"Extracting features from up to {max_images} images from the dataset...")
    count = 0
    for element in dataset.take(max_images):
        try:
            img_tensor = element['image']
            img_tensor = tf.image.resize(img_tensor, (224, 224))
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            img_tensor = tf.keras.applications.vgg16.preprocess_input(img_tensor)
            feature_vector = model.predict(img_tensor, verbose=0).flatten()
            features.append(feature_vector)
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} images...")
        except Exception as e:
            print(f"Error processing image from dataset: {e}")
            continue
    return np.array(features)

# --- Step 2: Training the Unsupervised Anomaly Detection Model ---
def train_detector(features):
    """
    Trains a One-Class SVM model on the extracted features.
    """
    print("Training One-Class SVM model...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # --- THIS IS THE KEY CHANGE ---
    # We are making the boundary much more flexible by increasing nu.
    # This allows for more variety in "real" photos.
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.3) # <-- CHANGED FROM 0.1 to 0.3
    
    model.fit(scaled_features)
    print("Training complete.")
    return model, scaler

# --- Main script execution ---
if __name__ == '__main__':
    print("Loading CelebA dataset from TensorFlow Datasets...")
    try:
        real_dataset = tfds.load(
            name='celeb_a',
            split='train',
            as_supervised=False
        )
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit()

    print("Loading VGG16 model...")
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    print("VGG16 model loaded.")

    real_features = extract_features(real_dataset, vgg_model)
    
    if real_features.size > 0:
        if_model, feature_scaler = train_detector(real_features)
        
        # --- NEW PART: SAVE THE MODELS ---
        print("Saving models to disk...")
        joblib.dump(if_model, 'svm_model.joblib')
        joblib.dump(feature_scaler, 'scaler.joblib')
        print("Models saved successfully as 'svm_model.joblib' and 'scaler.joblib'.")
        print("You can now run the 'backend_server.py'.")
        
    else:
        print("No features extracted. The dataset might be empty or a problem occurred.")