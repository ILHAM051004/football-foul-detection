# Football Foul Detection System

## Setup Instructions

### 1. Install Backend Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server

In the project directory, run:

```bash
python app.py
```

You should see output like:
```
 * Running on http://localhost:5000
```

The backend will be available at `http://localhost:5000`

### 3. Open the Web Interface

Open `index.html` in a web browser or serve it with a local server:

```bash
# Using Python 3
python -m http.server 8000

# Or using Python 2
python -m SimpleHTTPServer 8000
```

Then open `http://localhost:8000` in your browser.

## How to Use

1. Make sure the Flask backend is running (step 2 above)
2. Open the website in your browser
3. Click "Upload Video Pertandingan" and select a video file
4. Click "Analisis Video" button
5. Wait for processing (this may take a few moments for large videos)
6. View the prediction results showing:
   - **Prediction Label**: "Foul Tackle" or "Clean Tackle"
   - **Confidence Score**: Percentage confidence in the prediction
   - **Progress Bar**: Visual representation of confidence

## Model Details

- **Architecture**: CNN (MobileNetV2) + Bidirectional LSTM
- **Input Format**: Video clips with 16 frames each
- **Features Extracted**:
  - Visual features from MobileNetV2 (1280 dimensions)
  - Optical flow motion features (1 dimension)
- **Output**: Binary classification (Foul Tackle vs Clean Tackle)
- **Accuracy**: ~96%

## File Structure

```
├── index.html                 # Web interface
├── style.css                  # Styling
├── app.py                     # Flask backend server
├── model_offence.keras        # Trained model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── tackle-classification-mobilenet-lstm.ipynb  # Training notebook
```

## Architecture Explanation

### CNN (MobileNetV2)
- Extracts visual features from each frame
- Lightweight and efficient
- Pre-trained on ImageNet

### Optical Flow
- Captures motion between consecutive frames
- Computed using Farneback algorithm

### Bidirectional LSTM
- Analyzes temporal sequences of features
- Captures tackling patterns across frames
- Separate LSTM for each clip (2 clips per action)

### Classification
- Binary cross-entropy loss
- Sigmoid activation (outputs 0-1)
- Predictions: 0-0.5 = Clean Tackle, 0.5-1 = Foul Tackle

## Troubleshooting

### Backend not connecting
- Make sure `app.py` is running on port 5000
- Check firewall settings allow localhost:5000
- View browser console for error messages

### Model not loading
- Verify `model_offence.keras` exists in the project directory
- Check TensorFlow version compatibility
- Try reinstalling TensorFlow: `pip install --upgrade tensorflow`

### Prediction taking too long
- Large video files require more processing time
- This depends on your system specifications
- Typical processing: 30-60 seconds per video

### CORS errors
- Ensure Flask-CORS is installed: `pip install Flask-CORS`
- Backend should output "CORS enabled" on startup

## Requirements

- Python 3.8+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- GPU recommended for faster processing (optional)
- 2GB+ RAM for model inference

## Performance Tips

1. Use videos with clear tackling action
2. Video resolution should be 480p or higher
3. Video duration: 1-5 seconds typically works best
4. Supported formats: MP4, WebM, AVI, MOV

## Notes

- The system uses two clips per video for better temporal analysis
- Optical flow helps capture motion patterns
- LSTM processes 16 frames from each clip
- Model trained on SoccerNet dataset
