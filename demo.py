"""
==============================================
  Seat Detection Demo - SWS405 Project
  Available vs Occupied Seats Detection
  Model: YOLOv8n
==============================================

SETUP (run once in terminal):
    pip install ultralytics opencv-python

HOW TO RUN:
    python demo.py

IMPORTANT:
    - Make sure best.pt is in the same folder as this file
    - You can change the image path at the bottom of this file
"""

from ultralytics import YOLO
import cv2
import os
import sys


def run_demo(image_path, model_path="best.pt", confidence=0.4):
    """
    Run seat detection on a single image.
    
    Args:
        image_path: path to your test image
        model_path: path to best.pt (default: same folder)
        confidence: detection threshold 0.0-1.0 (default: 0.4)
    """

    # ── Check model exists ──────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found: {model_path}")
        print("Make sure best.pt is in the same folder as demo.py")
        sys.exit(1)

    # ── Check image exists ──────────────────────────────────────────
    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found: {image_path}")
        print("Make sure your test image path is correct")
        sys.exit(1)

    # ── Load model ──────────────────────────────────────────────────
    print("\n Loading model...")
    model = YOLO(model_path)
    print(" Model loaded successfully")

    # ── Run inference ───────────────────────────────────────────────
    print(f"\n Running detection on: {image_path}")
    results = model.predict(image_path, conf=confidence, save=True)

    # ── Print results ───────────────────────────────────────────────
    result = results[0]
    boxes = result.boxes

    print(f"\n{'='*50}")
    print(f"  DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"  Total seats detected: {len(boxes)}")

    available_count = 0
    occupied_count = 0

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence_score = float(box.conf[0])

        if class_name == "available":
            available_count += 1
        elif class_name == "occupied":
            occupied_count += 1

        print(f"  - {class_name.upper()} (confidence: {confidence_score:.2f})")

    print(f"\n  Available seats : {available_count}")
    print(f"  Occupied seats  : {occupied_count}")
    print(f"{'='*50}")

    # ── Show image with bounding boxes ──────────────────────────────
    print("\n Displaying result... (press any key to close)")
    annotated_image = result.plot()
    cv2.imshow("Seat Detection - Available vs Occupied", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Save result ─────────────────────────────────────────────────
    save_path = result.save_dir
    print(f"\n Result saved to: {save_path}")


# ══════════════════════════════════════════════════════════════════
#   CHANGE THIS TO YOUR TEST IMAGE PATH
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Option 1: hardcode your image path here
    IMAGE_PATH = "test_image.jpg"   # <-- change this to your image filename

    # Option 2: pass image path as command line argument
    # python demo.py my_image.jpg
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    # Run the demo
    run_demo(
        image_path=IMAGE_PATH,
        model_path="best.pt",   # make sure best.pt is in the same folder
        confidence=0.4
    )