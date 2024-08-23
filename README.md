# Facial Recognition System

## Overview
This Python script implements a real-time facial recognition system using OpenCV, dlib, and face_recognition libraries. The system is capable of:
* Detecting faces in a video stream.
* Encoding facial features for comparison.
* Comparing detected faces with a database of known faces.
* Displaying the recognized person's name on the video frame.

![image](https://github.com/user-attachments/assets/f9c70064-82b1-4483-ab18-3f08ad15ee57)


## Requirements
* Python 3.x
* OpenCV
* dlib
* face_recognition
* numpy
* pickle

## Installation
1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate  # For Windows
   ```
2. **Install required packages:**
   ```bash
    pip install opencv-python dlib face_recognition numpy pickle
   ```
3. **Download pre-trained models:**
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat
- Place them in the modelos directory.

## Usage
1. **Prepare a dataset:**
- Create a directory to store images of known individuals.
- Run the training script (not included) to generate trained_model.pkl.
2. **Run the script:**
  ```bash
    python facial_recognition.py
  ```
3. **Press 'q' to quit.**

## Code Structure

- shape_predictor: Loads the dlib shape predictor for facial landmarks.
- face_recognition_model: Loads the dlib face recognition model.
- load_trained_model: Loads the pre-trained model with known face encodings and IDs.
- capture_frames: Captures frames from the webcam.
- recognize_faces: Detects faces, calculates encodings, compares with known encodings, and displays results.
