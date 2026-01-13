🦺Construction Site Safety Detection using YOLOv10

This project implements a real-time computer vision system using YOLOv10 (CNN-based object detection) to monitor construction site safety compliance.
It detects whether workers are wearing safety helmets and safety vests and identifies safety violations from live CCTV or webcam video.

This is an industry-grade AI solution used in smart cities, construction companies, and industrial safety monitoring.

🚀 Business Problem

Construction sites are one of the most dangerous workplaces.
Accidents occur due to:

Workers not wearing helmets

Missing safety vests

Lack of real-time monitoring

Manual supervision is slow and unreliable.
This system provides automatic, real-time safety monitoring.

🧠 AI Solution

A YOLOv10 deep learning model is trained to detect:

Person

Helmet

No-Helmet

Safety Vest

The system:

Reads live video

Detects PPE compliance

Highlights violations instantly

📊 Dataset

This project uses a real construction-site dataset from Kaggle:

Safety Helmet & Vest Detection Dataset
https://www.kaggle.com/datasets/andrewmvd/helmet-detection

The dataset contains thousands of labeled images with bounding boxes for helmet, vest, and people.

⚠️ Dataset is not included in this repository due to size.

🔁 End-to-End Pipeline
Dataset → YOLOv10 Training → Model Evaluation → Model Export → Live Detection → API

📁 Project Structure
construction-safety-yolov10/
│
├── data/               # Dataset (not uploaded)
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── detect.py
│
├── api/
│   └── app.py          # FastAPI for live detection
│
├── models/
│   └── best.pt         # Trained YOLOv10 model
│
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation
pip install -r requirements.txt

🏋️ Train YOLOv10
from ultralytics import YOLO

model = YOLO("yolov10n.pt")
model.train(data="safety.yaml", epochs=50, imgsz=640)

🎥 Real-Time Detection
from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict(source=0, show=True)


This runs live webcam detection.

🌐 API for CCTV
uvicorn api.app:app --reload


Open:

http://127.0.0.1:8000/docs

🛠 Tech Stack

Python

YOLOv10

OpenCV

PyTorch

FastAPI

CNN

💼 Why This Project Matters

This project demonstrates:

Deep learning with CNN

Object detection

Real-time video AI

Industrial safety use case

Production-style deployment

This is not a toy project — it is a real industry solution.

👨‍💻 Author

Syed Sadath G
Machine Learning Engineer | Computer Vision | Deep Learning
