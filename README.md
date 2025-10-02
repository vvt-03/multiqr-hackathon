# multiqr-hackathon
MultiQR Hackathon Project
Project Overview

This project provides a complete pipeline for QR code detection, decoding, and classification from medicine pack images. It uses:

YOLOv8 for QR code detection.

OpenCV QRCodeDetector for decoding QR codes.

Setup Instructions
1. Clone the repository
git clone <your-repo-url>
cd multiqr-hackathon

2. Create and activate Python virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

3. Install dependencies 
pip install -r requirements.txt

Preparing Dataset

Place your training images in data/train/images/

Place YOLO .txt annotations in data/train/labels/

Place your test images in data/test/images/

You can use Roboflow or any auto-annotation method to generate .txt files in YOLO format.

Stage 1: Train YOLOv8 Detection Model
python train.py

ses YOLOv8 pre-trained model (yolov8n.pt) for fast training.

Training outputs saved in: runs/train/YOLOv8n/weights/best.pt.

You can modify train.py to change:

epochs

batch size

image size 

Stage 1: Run Detection
python infer.py --input data/test/images/ --output outputs/submission_detection_1.json

Detects QR codes in test images.

Generates JSON in the required format.