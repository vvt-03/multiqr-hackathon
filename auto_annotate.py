import cv2
import os
import glob

# Paths
IMAGE_DIR = "data/images/val"
LABEL_DIR = "data/labels/val"
os.makedirs(LABEL_DIR, exist_ok=True)

# QR Detector
qr_detector = cv2.QRCodeDetector()

for img_path in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png")):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        continue

    h, w, _ = img.shape

    # detectAndDecodeMulti for multiple QR codes
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img)

    label_path = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

    if retval and points is not None:
        with open(label_path, "w") as f:
            for pts in points:
                x_min = int(pts[:, 0].min())
                x_max = int(pts[:, 0].max())
                y_min = int(pts[:, 1].min())
                y_max = int(pts[:, 1].max())

                # YOLO format
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                bbox_w = (x_max - x_min) / w
                bbox_h = (y_max - y_min) / h

                f.write(f"0 {x_center} {y_center} {bbox_w} {bbox_h}\n")

        print(f"✅ Annotated {img_path} with {len(points)} QR(s)")

    else:
        # No QR detected → empty file
        open(label_path, "w").close()
        print(f"❌ No QR detected in {img_path}")
