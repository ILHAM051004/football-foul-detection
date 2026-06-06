from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and base model
try:
    # Try loading .keras format with safe_mode disabled
    model = tf.keras.models.load_model('model_offence.keras', safe_mode=False)
    print("Model loaded successfully from model_offence.keras")
except Exception as e:
    print(f"Error loading from .keras with safe_mode=False: {e}")
    print("Attempting to load H5 format...")
    try:
        # Try loading H5 format
        model = tf.keras.models.load_model('model_offence.h5', compile=False)
        print("Model loaded successfully from model_offence.h5")
    except Exception as e2:
        print(f"Error loading model from H5: {e2}")
        print("WARNING: Model failed to load. Check if model files exist and are compatible with TensorFlow 2.16+")
        model = None

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False

def extract_frames(video_path, n_frames=16):
    """Extract frames from video at uniform intervals"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        return np.zeros((16, 224, 224, 3), np.float32)
    
    segments = np.linspace(0, total, n_frames + 1, dtype=int)
    frames = []
    
    for i in range(n_frames):
        start, end = segments[i], segments[i + 1]
        frame_idx = np.random.randint(start, max(start + 1, end))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            frames.append(np.zeros((224, 224, 3), np.float32))
            continue
        
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames, np.float32)

def compute_motion(frames):
    """Compute optical flow between frames"""
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(np.mean(flow))
        prev = curr
    
    flows = np.array(flows + [0.0])
    return flows.reshape(-1, 1)

def extract_features(frames, base_model):
    """Extract features using MobileNetV2"""
    frames_preprocessed = preprocess_input(frames.astype(np.float32))
    return base_model.predict(frames_preprocessed, verbose=0)

def process_video(video_path):
    """Process video and return model prediction"""
    try:
        # Extract frames
        frames = extract_frames(video_path)
        
        # Extract CNN features using MobileNetV2
        frame_feat = extract_features(frames, base_model)  # (16, 1280)
        
        # Compute motion features
        motion_feat = compute_motion(frames)  # (16, 1)
        
        # Combine features
        combined = np.concatenate([frame_feat, motion_feat], axis=1)  # (16, 1281)
        
        # Prepare input for model (2 clips with same features for now)
        # Input shape: (1, 2, 16, 1281)
        input_data = np.stack([combined, combined], axis=0)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Make prediction
        prediction = model.predict(input_data, verbose=0)[0][0]
        
        # Convert prediction to label and confidence
        confidence = float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100
        label = "Offence" if prediction > 0.5 else "No Offence"
        
        return {
            "success": True,
            "prediction": float(prediction),
            "label": label,
            "confidence": confidence,
            "display_label": "Foul Tackle" if label == "Offence" else "Clean Tackle"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def process_dual_video(video_path_live, video_path_highlight):
    """Process two videos (live and highlight) and return model prediction
    
    If a video is not provided, it's filled with zeros.
    clip_0 = live video (or zeros if not provided)
    clip_1 = highlight video (or zeros if not provided)
    """
    try:
        # Initialize clips with zeros
        clip_0_features = np.zeros((16, 1281), np.float32)  # clip_0 = live video
        clip_1_features = np.zeros((16, 1281), np.float32)  # clip_1 = highlight video
        
        # Process live video if provided
        if video_path_live:
            frames_live = extract_frames(video_path_live)
            frame_feat_live = extract_features(frames_live, base_model)  # (16, 1280)
            motion_feat_live = compute_motion(frames_live)  # (16, 1)
            clip_0_features = np.concatenate([frame_feat_live, motion_feat_live], axis=1)  # (16, 1281)
        
        # Process highlight video if provided
        if video_path_highlight:
            frames_highlight = extract_frames(video_path_highlight)
            frame_feat_highlight = extract_features(frames_highlight, base_model)  # (16, 1280)
            motion_feat_highlight = compute_motion(frames_highlight)  # (16, 1)
            clip_1_features = np.concatenate([frame_feat_highlight, motion_feat_highlight], axis=1)  # (16, 1281)
        
        # Stack as two clips [clip_0, clip_1]
        # Input shape: (1, 2, 16, 1281)
        input_data = np.stack([clip_0_features, clip_1_features], axis=0)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Make prediction
        prediction = model.predict(input_data, verbose=0)[0][0]
        
        # Convert prediction to label and confidence
        confidence = float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100
        label = "Offence" if prediction > 0.5 else "No Offence"
        
        return {
            "success": True,
            "prediction": float(prediction),
            "label": label,
            "confidence": confidence,
            "display_label": "Foul Tackle" if label == "Offence" else "Clean Tackle"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Handle video upload and prediction"""
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    
    video_file_live = request.files.get('video_live')
    video_file_highlight = request.files.get('video_highlight')
    
    # At least one video is required
    if not video_file_live and not video_file_highlight:
        return jsonify({"success": False, "error": "At least one video file is required"}), 400
    
    try:
        temp_files = []
        video_path_live = None
        video_path_highlight = None
        
        # Save live video if provided
        if video_file_live and video_file_live.filename != '':
            temp_file_live = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            video_file_live.save(temp_file_live.name)
            temp_file_live.close()
            video_path_live = temp_file_live.name
            temp_files.append(temp_file_live.name)
        
        # Save highlight video if provided
        if video_file_highlight and video_file_highlight.filename != '':
            temp_file_highlight = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            video_file_highlight.save(temp_file_highlight.name)
            temp_file_highlight.close()
            video_path_highlight = temp_file_highlight.name
            temp_files.append(temp_file_highlight.name)
        
        # Process videos
        result = process_dual_video(video_path_live, video_path_highlight)
        
        # Clean up
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
