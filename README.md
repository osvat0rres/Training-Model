# YOLO Object Detection Project

This project contains files for training and running object detection models using the YOLO (You Only Look Once) algorithm. The models can be trained using either a CPU or GPU, with GPU training providing significantly faster performance.

The trained model in this project is focused on detecting common household objects.

---

## Features

- Train YOLO object detection models
- Use a CPU or GPU for training
- Detect objects in:
  - Images
  - Videos
  - Live webcam feeds

---

## Project Files

### Training Files
These files are used to train the YOLO model on custom datasets.

- Dataset configuration and training scripts
- Supports both CPU and GPU training
- GPU is recommended for faster training times

---

## Detection Files

### `imageOutput.py`
Uses the trained YOLO model to detect objects in a single image.

**Example use cases:**
- Detect household items in photos
- Test model accuracy on images

---

### `videoOutput.py`
Uses the trained model to detect objects frame-by-frame in a video.

**Example use cases:**
- Analyze recorded footage
- Perform object tracking in videos

---

### `webcamOutput.py`
Runs real-time object detection using the computer’s webcam.

**Example use cases:**
- Live object detection
- Real-time testing of the trained model

---

## Virtual Environment Setup

It is recommended to create and activate a Python virtual environment before installing dependencies.

### Create a virtual environment
```bash
python -m venv venv
```

### Activate the virtual environment

#### Windows
```bash
venv\Scripts\activate
```

#### macOS / Linux
```bash
source venv/bin/activate
```

---

## Requirements

Some common libraries used in this project may include:

- Python 3
- OpenCV
- Ultralytics YOLO
- PyTorch

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run image detection
```bash
python imageOutput.py
```

### Run video detection
```bash
python videoOutput.py
```

### Run webcam detection
```bash
python webcamOutput.py
```

---

## Notes

- The current trained model is designed for household object detection.
- Make sure the trained YOLO model weights are correctly placed in the project directory before running detection scripts.
- Webcam detection requires a working camera connected to the device.
- Using a GPU is recommended for faster training and detection performance.

