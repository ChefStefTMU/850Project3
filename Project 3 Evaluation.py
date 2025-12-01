import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/project_3_model/weights/best.pt")

# Input images
image_paths = [
    "data/data/evaluation/ardmega.jpg",
    "data/data/evaluation/arduno.jpg",
    "data/data/evaluation/rasppi.jpg"
]

# Output directory
output_dir = "AER850_Project3/evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Class → Color map
color_map = {
    "connector":  (0, 140, 255),
    "ic":         (0, 255, 0),
    "capacitor":  (0, 255, 255),
    "resistor":   (255, 180, 0),
    "diode":      (0, 200, 255),
    "led":        (0, 255, 150),
    "button":     (255, 128, 0)
}

# Process each evaluation image
for img_path in image_paths:

    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        continue

    h, w = img.shape[:2]
    base = w / 2000
    font_scale = max(0.4, 0.7 * base)
    thickness = max(1, int(2 * base))

    # YOLO prediction
    results = model(img, conf=0.25, iou=0.7)[0]

    # Draw bounding boxes
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results.names[cls].lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        text = f"{label} {conf:.2f}"

        # Determine color
        color = (0, 150, 255)
        for key in color_map:
            if key in label:
                color = color_map[key]
                break

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 10, y1), color, -1)

        # Text
        cv2.putText(img, text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Save output image
    out_name = os.path.basename(img_path).replace(".jpg", "_pred.jpg")
    save_path = os.path.join(output_dir, out_name)
    cv2.imwrite(save_path, img)

    print("Saved prediction:", save_path)
