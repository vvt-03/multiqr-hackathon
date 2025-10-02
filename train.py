from ultralytics import YOLO


model = YOLO("yolov8n.pt")  # 'n' for nano (fast)
model.train(data="qr_data.yaml", epochs=50, imgsz=640, batch=8)
