# Quick Start Guide

## Windows Users - Easy Setup

### Step 1: Install Python Dependencies
Double-click `run_backend.bat` to automatically install required packages and start the server.

Alternatively, open Command Prompt in this folder and run:
```
python -m pip install -r requirements.txt
```

### Step 2: Start Backend Server
Double-click `run_backend.bat` to start the Flask server.

You should see:
```
 * Running on http://127.0.0.1:5000
```

Keep this window open while using the system.

### Step 3: Open Web Interface
Option A - Using Python's built-in server:
```
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser

Option B - Direct file opening:
Simply double-click `index.html` to open it directly in your browser

## Mac/Linux Users

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Backend Server
```bash
python app.py
```

### Step 3: Open Web Interface
```bash
# In a new terminal
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser

## Quick Test

1. After both servers are running:
   - Backend: `http://localhost:5000` (Flask)
   - Frontend: `http://localhost:8000` or direct `index.html`

2. Select a video file
3. Click "Analisis Video"
4. Wait for results

## Troubleshooting

- **"Port 5000 already in use"**: Another program is using port 5000
  - Kill the process or edit `app.py` to use a different port

- **"Model not loaded"**: `model_offence.keras` file missing
  - Ensure the file exists in the project directory

- **Connection error**: Frontend can't reach backend
  - Make sure backend is running at `http://localhost:5000`
  - Check firewall settings
  - Try opening `http://localhost:5000/health` in browser

- **Slow processing**: Normal behavior for video inference
  - Typical processing: 30-60 seconds per video
  - Depends on video length and system specs
