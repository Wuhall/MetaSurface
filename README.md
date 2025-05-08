# Metal Surface Defect Detection API

This project provides a Flask-based API for detecting defects on metal surfaces using a YOLO model. The API supports image uploads and returns detection results in JSON format.

## Features

- **Object Detection**: Detects objects (e.g., helmets, defects) in images using YOLO.
- **Image Upload**: Supports image uploads via a web interface or API.
- **Real-Time Predictions**: Streams predictions for uploaded images.
- **Customizable Models**: Easily switch between different YOLO models.

### Key Files and Directories

- **`predict_api2.py`**: Main Flask application for handling API requests.
- **`run_app.sh`**: Script to start the application with specific weights and classes.
- **`weights/`**: Directory containing YOLO model weights.
- **`templates/`**: HTML templates for the web interface.
- **`uploads/`**: Directory for storing uploaded images.
- **`utils/general.py`**: Utility functions for the project.

## Installation

1. Clone the repository
2. Install dependencies
```
conda create --name meta-surface python=3.9 -y
conda activate meta-surface
pip install -r requirements.txt
```

## Run
```
python predict_api2.py --weight ./weights/metalsurface_best.pt  --classes 0
```


