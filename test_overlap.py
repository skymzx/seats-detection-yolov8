"""
==============================================================
   PERSON-CHAIR OVERLAP DETECTION
   No training required - uses YOLOv8 COCO pretrained model
==============================================================

HOW IT WORKS:
  1. Detect all people and chairs in the image (using COCO model)
  2. For each chair, check if any person overlaps it
  3. If overlap > threshold → chair is OCCUPIED
  4. If no overlap → chair is AVAILABLE

WHY THIS WORKS BETTER:
  Instead of training "what does an occupied chair look like?"
  (which is hard because people block chairs differently),
  we use what's already easy:
  - "Where are the people?" (well-known COCO class)
  - "Where are the chairs?" (well-known COCO class)
  Then combine them with overlap math.

HOW TO RUN:
  python test_overlap.py path/to/your/image.jpg
  
  Or just:
  python test_overlap.py
  (uses default test image)
"""

from ultralytics import YOLO
import cv2
import sys
import os


# ==============================================================
#  CONFIGURATION
# ==============================================================

# Use any YOLOv8 model pretrained on COCO (no custom training needed)
MODEL = "yolov8m.pt"   # m for better accuracy, or yolov8n.pt for speed

# COCO class IDs
PERSON_CLASS = 0
CHAIR_CLASS  = 56

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.3       # min confidence for person/chair
OVERLAP_THRESHOLD    = 0.15      # min IoU between person and chair to count as occupied


# ==============================================================
#  IOU CALCULATION
# ==============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # No intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def chair_overlap_ratio(chair_box, person_box):
    """
    Calculate how much of the CHAIR is covered by the person.
    Different from IoU - measures person-on-chair occupancy.
    """
    x1 = max(chair_box[0], person_box[0])
    y1 = max(chair_box[1], person_box[1])
    x2 = min(chair_box[2], person_box[2])
    y2 = min(chair_box[3], person_box[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    chair_area = (chair_box[2] - chair_box[0]) * (chair_box[3] - chair_box[1])
    
    return intersection / chair_area if chair_area > 0 else 0.0


# ==============================================================
#  DETECT AND CLASSIFY
# ==============================================================

def detect_seats(image_path):
    print("=" * 70)
    print("  PERSON-CHAIR OVERLAP DETECTION")
    print("=" * 70)
    print(f"\n   Image:     {image_path}")
    print(f"   Model:     {MODEL} (COCO pretrained)")
    print(f"   Conf:      {CONFIDENCE_THRESHOLD}")
    print(f"   Overlap:   {OVERLAP_THRESHOLD}")
    print()
    
    # Load model
    model = YOLO(MODEL)
    
    # Run inference
    results = model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    
    # Separate persons and chairs
    persons = []
    chairs = []
    
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        if cls == PERSON_CLASS:
            persons.append({
                "box":  xyxy,
                "conf": conf
            })
        elif cls == CHAIR_CLASS:
            chairs.append({
                "box":  xyxy,
                "conf": conf,
                "status": "available"   # default
            })
    
    print(f"   Found {len(persons)} people")
    print(f"   Found {len(chairs)} chairs\n")
    
    # Check overlap for each chair
    available_count = 0
    occupied_count = 0
    
    for i, chair in enumerate(chairs):
        max_overlap = 0
        for person in persons:
            overlap = chair_overlap_ratio(chair["box"], person["box"])
            if overlap > max_overlap:
                max_overlap = overlap
        
        if max_overlap > OVERLAP_THRESHOLD:
            chair["status"] = "occupied"
            chair["overlap"] = max_overlap
            occupied_count += 1
        else:
            chair["overlap"] = max_overlap
            available_count += 1
        
        print(f"   Chair {i+1}: {chair['status'].upper():10s} "
              f"(conf={chair['conf']:.2f}, max overlap with person={max_overlap:.2f})")
    
    print(f"\n   SUMMARY: {available_count} available, {occupied_count} occupied\n")
    
    return persons, chairs


# ==============================================================
#  VISUALIZE RESULTS
# ==============================================================

def visualize(image_path, persons, chairs, save_path="overlap_result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Couldn't read image: {image_path}")
        return
    
    # Draw chairs
    for chair in chairs:
        x1, y1, x2, y2 = [int(v) for v in chair["box"]]
        
        if chair["status"] == "occupied":
            color = (0, 100, 255)   # orange
            label = f"OCCUPIED ({chair['overlap']:.0%})"
        else:
            color = (0, 255, 0)     # green
            label = f"AVAILABLE"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw persons (lighter, for context)
    for person in persons:
        x1, y1, x2, y2 = [int(v) for v in person["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 100), 1)
        cv2.putText(img, "person", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    # Save and show
    cv2.imwrite(save_path, img)
    print(f"✅ Result saved to: {save_path}\n")
    
    # Display
    cv2.imshow("Seat Detection - Overlap Method", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default test image if available
        candidates = [
            "test_image.jpg",
            "Seats-Detection-4/test/images",   # folder - pick first image
        ]
        
        image_path = None
        for c in candidates:
            if os.path.isfile(c):
                image_path = c
                break
            elif os.path.isdir(c):
                images = [f for f in os.listdir(c) if f.endswith(('.jpg', '.png'))]
                if images:
                    image_path = os.path.join(c, images[0])
                    break
        
        if not image_path:
            print("❌ No image specified and no test image found.")
            print("   Usage: python test_overlap.py path/to/image.jpg")
            sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    persons, chairs = detect_seats(image_path)
    visualize(image_path, persons, chairs)
