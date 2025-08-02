
# 🔭 AI Space Station Object Detection – Duality AI Hackathon

## 🚀 Project Overview

This project focuses on training an object detection model using synthetic data generated from Duality AI's Falcon platform. The objective is to detect and classify three object types—**Toolbox**, **Oxygen Tank**, and **Fire Extinguisher**—within a simulated space station environment using the **YOLOv8** model.

---

## 📁 Folder Structure

```
AI_Space_duality/
├── train.py
├── predict.py
├── config.yaml
├── runs/
│   ├── train/
│   └── val/
├── AI_Space_duality.ipynb
├── README.md
└── report.pdf
```

---



## 🏋️‍♂️ Training the Model

1. The model has been tranied using Google Colab.

---

### Outputs include:
- `mAP@0.5`
- Precision & Recall
- Confusion Matrix
- Annotated images with predicted bounding boxes

---

## 📈 Results

| Metric           | Score     |
|------------------|-----------|
| mAP@0.5          | _98%_ |
| Precision        | _91%_ |
| Recall           | _0.94359_ |

Confusion matrix and visual samples are included in the `runs/` folder and final report.

---

## 🧠 Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Occlusion of Oxygen Tanks | Added data augmentation with occlusion simulation |
| Class imbalance | Applied resampling and weighted loss |
| Overfitting | Early stopping + dropout regularization |

---

## Use Case Application 

An application was developed for desktop using the trained model to detect and classify objects in real-time.

---

## 🌐 Live Demo

The application is live and accessible here:
👉 [AI-Powered Object Detection System](https://ai-powered-obj-detection-system.streamlit.app/)


---

# AI-powered-Obj-Detection-System
This repo Consist of AI-powered Object detection system for space station environment


# YOLOv8 Detection App

This is a simple Streamlit web application for real-time object detection using the YOLOv8 model. It supports both **webcam streaming** and **video file uploads** for detecting objects.

---

## 🚀 Features

- **Webcam Mode**: Live object detection from your webcam.
- **Upload Mode**: Upload a video file (mp4, avi, mov, mkv) for batch processing.
- **Confidence Threshold**: Adjustable slider to control detection confidence.
- Built on [Ultralytics YOLOv8](https://docs.ultralytics.com/), Streamlit, and OpenCV.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.7 or later.

---

## 🛠️ Usage

1. Place your YOLOv8 model weights as `best.pt` in the same directory as `app.py`.
2. Run the app:

```bash
streamlit run app.py
```

3. Select **Webcam** or **Upload Video** mode from the sidebar.
4. Adjust the confidence threshold as needed.


---

## 📌 Notes

- The app uses `ultralytics.YOLO` for inference. Make sure `best.pt` is compatible with YOLOv8.
- Streamlit WebRTC may require webcam permissions in your browser.
- Tested on Google Chrome and Firefox.

---

## 🧠 Author

Made with Team Graviton using Streamlit and YOLOv8.

---








