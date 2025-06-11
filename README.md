# 🖐️ Real-Time Sign Language Recognition using MediaPipe & Neural Networks
📌 Overview
This project bridges the communication gap between sign language users and non-users by recognizing hand gestures in real time using hand landmark detection and machine learning. Leveraging MediaPipe by Google and a trained Neural Network, the system identifies sign language gestures from webcam input and displays the recognized text on the screen — enabling a seamless, interactive communication experience.

🎯 Features
🔍 Real-time hand gesture recognition via webcam

📐 21-point hand landmark detection using MediaPipe

🤖 Gesture classification using a trained Neural Network

📊 Dataset generated and stored in CSV format for training

📺 Live prediction display on screen

🛠️ Tech Stack
MediaPipe – for real-time hand landmark detection

OpenCV – for webcam frame capture and processing

Python – core programming language

NumPy & Pandas – for data manipulation

TensorFlow/Keras – to build and train the neural network

🧠 How It Works
Landmark Detection
MediaPipe extracts 21 hand landmarks per frame (x, y, z coordinates).

Dataset Creation
Thousands of samples per gesture are captured and saved as rows in a CSV file.

Model Training
A neural network is trained on this structured dataset to classify gestures accurately.

Real-Time Prediction
Webcam input is analyzed frame by frame, with predicted gesture text shown on the screen.
