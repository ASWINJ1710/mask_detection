# Real-Time Face Mask Detection System using opencv & streamlit

<p align="center">
  A Computer Vision Application for Real-Time Face Mask Detection
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 1. Overview

The **Real-Time Face Mask Detection System** is a computer vision application that detects human faces and identifies whether individuals are wearing protective face masks.

The application combines traditional image processing techniques with deep learning-based image classification to perform face mask detection through a live webcam feed.

It implements a two-stage processing architecture:

- **Face Detection:** OpenCV Haar Cascade identifies and localizes human faces.
- **Mask Classification:** A trained MobileNetV2-based neural network classifies each detected face into one of two categories: `With Mask` or `Without Mask`.

A Streamlit-based dashboard provides an interactive interface for accessing the webcam, monitoring detection results, adjusting the classification threshold, and viewing real-time performance statistics.

## 2. Problem Statement

Manual monitoring of face mask usage in public environments can be time-consuming and inefficient.

This project addresses the problem by developing an automated vision-based system capable of detecting multiple faces and identifying mask usage from live video streams.

The objective is to provide an interactive and efficient solution that combines face localization, image classification, and real-time visualization.

## 3. Project Objectives

- Develop an automated face detection and mask classification system.
- Integrate OpenCV-based face detection with a trained deep learning classifier.
- Enable real-time webcam processing.
- Identify multiple faces within a single frame.
- Display classification results with bounding boxes and confidence scores.
- Provide adjustable prediction thresholds.
- Develop an interactive dashboard using Streamlit.
- Monitor live detection statistics and processing FPS.

## 4. Technology Stack

| Technology | Application |
|---|---|
| Python | Application development |
| OpenCV | Image processing and face detection |
| Haar Cascade | Face localization |
| MobileNetV2 | Mask classification |
| TensorFlow / Keras | Model development and inference |
| Streamlit | Interactive dashboard |
| Streamlit-WebRTC | Real-time webcam streaming |
| NumPy | Numerical processing |
| PyAV | Video frame handling |

## 5. System Architecture

The application follows a two-stage computer vision pipeline.

### Stage 1: Face Detection

OpenCV Haar Cascade is responsible for identifying human faces in incoming video frames.

Processing operations:

1. Capture the webcam frame.
2. Convert the frame from BGR to grayscale.
3. Apply histogram equalization.
4. Execute Haar Cascade face detection.
5. Extract face coordinates using bounding boxes.

### Stage 2: Face Mask Classification

The detected face regions are passed to a trained MobileNetV2-based classification model.

Processing operations:

1. Extract the detected face region.
2. Expand the crop slightly using a margin.
3. Convert the image from BGR to RGB.
4. Resize the face image to 224 × 224 pixels.
5. Perform model inference.
6. Calculate the predicted class and confidence.
7. Generate the final classification result.

### Architecture Diagram

```text
              WEBCAM INPUT
                    |
                    v
          STREAMLIT-WEBRTC
                    |
                    v
             VIDEO FRAME
                    |
                    v
          OPENCV PREPROCESSING
                    |
                    v
          HAAR CASCADE DETECTOR
                    |
                    v
             FACE REGIONS
                    |
                    v
          FACE CROP & RESIZE
                    |
                    v
          MOBILENETV2 MODEL
                    |
                    v
           MASK CLASSIFICATION
                    |
             +------+------+
             |             |
             v             v
          WITH MASK     WITHOUT MASK
             |             |
             v             v
         GREEN BOX      RED BOX
             |             |
             +------+------+
                    |
                    v
          ANNOTATED VIDEO FRAME
                    |
                    v
          STREAMLIT DASHBOARD
                    |
                    v
           LIVE STATISTICS
```

## 6. Model Information

### Face Detection Model

**Algorithm:** Haar Cascade Classifier

Haar Cascade uses Haar-like features and a cascade of classifiers to identify frontal human faces.

Key configuration:

- Scale Factor: 1.1
- Minimum Neighbors: 5
- Minimum Face Size: 60 × 60 pixels
- Input: Grayscale image

### Face Mask Classification Model

**Architecture:** MobileNetV2

MobileNetV2 is a lightweight convolutional neural network architecture designed for computational efficiency.

The trained model is used to classify detected face images into two categories.

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV2-based classifier |
| Input Size | 224 × 224 |
| Input Channels | 3 (RGB) |
| Output | Binary classification |
| Class 1 | With Mask |
| Class 2 | Without Mask |
| Activation | Model-defined |

The trained model is stored in:

```text
models/face_mask_model.keras
```

## 7. Application Features

### Real-Time Webcam Detection

Processes live webcam frames through the browser-based Streamlit-WebRTC interface.

### Face Localization

Identifies face regions and displays bounding boxes around detected individuals.

### Mask Classification

Classifies each detected face into:

- With Mask
- Without Mask

### Confidence Score

Displays the model's prediction confidence alongside the classification label.

### Adjustable Decision Threshold

Allows users to modify the classification decision threshold through the Streamlit sidebar.

### Live Performance Dashboard

Displays continuously updated statistics:

| Metric | Description |
|---|---|
| Faces in Frame | Total classified faces |
| With Mask | Number of masked faces |
| Without Mask | Number of unmasked faces |
| FPS | Estimated frame processing rate |

### Visual Classification

- Green bounding box – With Mask
- Red bounding box – Without Mask

## 8. Project Directory Structure

```text
mask_detection/
│
├── models/
│   └── face_mask_model.keras
│
├── src/
│   ├── face_detector.py
│   ├── mask_classifier.py
│   └── video_processor.py
│
├── training/
│
├── results/
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or compatible version
- Visual Studio Code
- Webcam access
- Internet connection for initial dependency installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ASWINJ1710/mask_detection.git
```

### Step 2: Navigate to the Project Directory

```bash
cd mask_detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Model Availability

Ensure that the trained model is available at:

```text
models/face_mask_model.keras
```

The application requires this model for mask classification.

### Step 7: Launch the Application

```bash
streamlit run app.py
```

The application will launch in your default browser.

Allow webcam access when prompted and start the live video processing session.

## 10. Detection Results

The system produces annotated video frames containing:

- Detected face bounding boxes.
- Predicted mask classification.
- Prediction confidence percentage.
- Live detection statistics.

**Color representation:**

| Color | Classification |
|---|---|
| Green | With Mask |
| Red | Without Mask |

### Application Screenshots

Add your actual Streamlit dashboard screenshots here.

```markdown
![Dashboard](results/dashboard.png)

![Mask Detection](results/with_mask.png)

![No Mask Detection](results/without_mask.png)
```

## 11. Performance Considerations

The system's performance depends on:

- Webcam resolution and quality.
- Lighting conditions.
- Number of faces in the frame.
- Face detection parameters.
- Available CPU/GPU resources.
- Classification model inference time.

The application reports an estimated FPS value to provide visibility into live processing performance.

No fixed accuracy or FPS value is claimed without measured evaluation results.

## 12. Limitations

- Haar Cascade primarily targets frontal faces.
- Extreme face angles may reduce detection reliability.
- Poor lighting can affect face localization.
- Small or partially occluded faces may be missed.
- Classification performance depends on training dataset quality.
- The current classifier supports only two categories.
- The system does not explicitly identify incorrectly worn masks.

## 13. Future Enhancements

- Introduce more robust face detection algorithms such as OpenCV YuNet.
- Add an incorrect-mask classification category.
- Improve model generalization through dataset augmentation.
- Add image and video upload functionality.
- Introduce face tracking for consistent identification across frames.
- Implement detection history and downloadable reports.
- Optimize inference performance for low-resource systems.

## 14. Conclusion

The Real-Time Face Mask Detection System demonstrates the integration of computer vision, deep learning, and interactive web application development.

By combining OpenCV Haar Cascade face detection with a trained MobileNetV2-based classifier, the application provides a practical approach to identifying face mask usage through live webcam input.

The Streamlit interface makes the system accessible and interactive, while real-time classification, visual annotations, confidence scores, and performance statistics provide a comprehensive demonstration of the implemented pipeline.

The project also establishes a foundation for further improvements in real-time face analysis and computer vision applications.

---

## 15. Author

**ASWIN J**

GitHub: [@ASWINJ1710](https://github.com/ASWINJ1710)

**Project Repository:**  
[Real-Time Face Mask Detection System](https://github.com/ASWINJ1710/mask_detection)

---

<p align="center">
  Developed as a Computer Vision and Deep Learning Project.
</p>
