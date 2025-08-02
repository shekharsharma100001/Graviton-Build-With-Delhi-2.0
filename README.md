
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




