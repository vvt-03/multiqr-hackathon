import json
from ultralytics import YOLO
import glob
import os
import argparse

def main(input_folder, output_file):
    # Load your trained model
    model = YOLO("runs/detect/train3/weights/best.pt")  # replace with your best.pt path
    
    # Get all images in the folder
    image_files = glob.glob(os.path.join(input_folder, "*"))
    image_files = [f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    submission = []

    for img_path in image_files:
        results = model(img_path)[0]  # run inference
        boxes = results.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            submission.append({
                "image_id": os.path.basename(img_path),
                "category_id": 1,  # QR code class
                "bbox": bbox,
                "score": float(score)
            })

    # Save results as JSON
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=4)

    print(f"Inference complete. {len(submission)} detections saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO QR code detection")
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    main(args.input, args.output)
