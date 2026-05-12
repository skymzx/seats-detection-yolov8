"""
==============================================================
   EVALUATE OVERLAP METHOD ON ENTIRE DATASET
==============================================================

WHAT THIS DOES:
  1. Runs the person-chair overlap method on all test images
  2. Counts how many chairs were detected as available/occupied
  3. Compares predictions to ground truth labels
  4. Calculates Precision, Recall, and Accuracy
  5. Saves a CSV with per-image results
  6. Saves a summary report

WHY THIS MATTERS:
  This tells you objectively how well the overlap method works
  across your entire dataset, not just a few cherry-picked images.

HOW TO RUN:
  python evaluate_overlap.py
"""

from ultralytics import YOLO
import os
import csv
from collections import defaultdict


# ==============================================================
#  CONFIGURATION
# ==============================================================

MODEL = "yolov8m.pt"             # COCO pretrained, no custom training
TEST_IMAGES_DIR = "Seats-Detection-4/test/images"
TEST_LABELS_DIR = "Seats-Detection-4/test/labels"

# COCO class IDs
PERSON_CLASS = 0
CHAIR_CLASS  = 56

# Your custom dataset class IDs (from data.yaml)
# Check data.yaml - usually:  0 = available, 1 = occupied
AVAILABLE_CLASS_ID = 0
OCCUPIED_CLASS_ID  = 1

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.25
OVERLAP_THRESHOLD    = 0.15


# ==============================================================
#  IOU HELPERS
# ==============================================================

def chair_overlap_ratio(chair_box, person_box):
    """How much of the chair is covered by the person"""
    x1 = max(chair_box[0], person_box[0])
    y1 = max(chair_box[1], person_box[1])
    x2 = min(chair_box[2], person_box[2])
    y2 = min(chair_box[3], person_box[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    chair_area = (chair_box[2] - chair_box[0]) * (chair_box[3] - chair_box[1])
    return intersection / chair_area if chair_area > 0 else 0.0


def iou(box1, box2):
    """Standard IoU for matching predictions to ground truth"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - intersection
    return intersection / union if union > 0 else 0.0


# ==============================================================
#  PARSE YOLO LABELS
# ==============================================================

def load_ground_truth(label_path, img_width, img_height):
    """
    Load YOLO format labels.
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    Returns list of {class_id, box} where box is [x1, y1, x2, y2] in pixels
    """
    if not os.path.exists(label_path):
        return []
    
    ground_truth = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width    = float(parts[3]) * img_width
            height   = float(parts[4]) * img_height
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            ground_truth.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2]
            })
    
    return ground_truth


# ==============================================================
#  PREDICT WITH OVERLAP METHOD
# ==============================================================

def predict_overlap(model, image_path):
    """
    Returns list of chair predictions with status (available/occupied)
    """
    results = model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    
    persons = []
    chairs = []
    
    for box in result.boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        if cls == PERSON_CLASS:
            persons.append(xyxy)
        elif cls == CHAIR_CLASS:
            chairs.append({
                "box": xyxy,
                "status": "available",
                "overlap": 0
            })
    
    # Classify each chair
    for chair in chairs:
        max_overlap = 0
        for person in persons:
            ov = chair_overlap_ratio(chair["box"], person)
            if ov > max_overlap:
                max_overlap = ov
        
        chair["overlap"] = max_overlap
        if max_overlap > OVERLAP_THRESHOLD:
            chair["status"] = "occupied"
    
    return persons, chairs


# ==============================================================
#  MATCH PREDICTIONS TO GROUND TRUTH
# ==============================================================

def evaluate_image(predictions, ground_truth):
    """
    Match predicted chairs to ground truth chairs using IoU.
    Count true positives, false positives, false negatives per class.
    """
    matched_gt = set()
    
    # Track results per class
    stats = {
        "available": {"tp": 0, "fp": 0, "fn": 0},
        "occupied":  {"tp": 0, "fp": 0, "fn": 0},
    }
    
    # For each prediction, find best matching ground truth
    for pred in predictions:
        best_iou = 0
        best_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            ov = iou(pred["box"], gt["box"])
            if ov > best_iou:
                best_iou = ov
                best_idx = i
        
        # IoU > 0.2 = match (lowered from 0.5 because COCO chair boxes and your
        # custom annotations have different conventions for occupied seats)
        if best_iou > 0.2 and best_idx >= 0:
            matched_gt.add(best_idx)
            gt = ground_truth[best_idx]
            gt_class = "available" if gt["class_id"] == AVAILABLE_CLASS_ID else "occupied"
            pred_class = pred["status"]
            
            if pred_class == gt_class:
                stats[pred_class]["tp"] += 1
            else:
                stats[pred_class]["fp"] += 1
                stats[gt_class]["fn"] += 1
        else:
            # False positive - predicted a chair that doesn't exist
            stats[pred["status"]]["fp"] += 1
    
    # False negatives - ground truth chairs we missed
    for i, gt in enumerate(ground_truth):
        if i not in matched_gt:
            gt_class = "available" if gt["class_id"] == AVAILABLE_CLASS_ID else "occupied"
            stats[gt_class]["fn"] += 1
    
    return stats


# ==============================================================
#  MAIN EVALUATION LOOP
# ==============================================================

def evaluate_dataset():
    print("=" * 70)
    print("  EVALUATING OVERLAP METHOD ON TEST SET")
    print("=" * 70)
    print(f"\n   Model:               {MODEL} (COCO pretrained)")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Overlap threshold:    {OVERLAP_THRESHOLD}")
    print(f"   Test images:          {TEST_IMAGES_DIR}\n")
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Test folder not found: {TEST_IMAGES_DIR}")
        return
    
    # Load model
    model = YOLO(MODEL)
    
    # Get all images
    images = [f for f in os.listdir(TEST_IMAGES_DIR) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"   Found {len(images)} test images\n")
    
    # Aggregate stats
    total_stats = {
        "available": {"tp": 0, "fp": 0, "fn": 0},
        "occupied":  {"tp": 0, "fp": 0, "fn": 0},
    }
    
    per_image_results = []
    
    # Process each image
    import cv2
    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(TEST_LABELS_DIR, label_name)
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Get predictions
        persons, chairs = predict_overlap(model, img_path)
        
        # Get ground truth
        ground_truth = load_ground_truth(label_path, w, h)
        
        # Evaluate
        stats = evaluate_image(chairs, ground_truth)
        
        # Aggregate
        for cls in total_stats:
            for metric in total_stats[cls]:
                total_stats[cls][metric] += stats[cls][metric]
        
        # Save per-image
        per_image_results.append({
            "image": img_name,
            "num_persons": len(persons),
            "num_chairs_pred": len(chairs),
            "num_gt_objects": len(ground_truth),
            "available_tp": stats["available"]["tp"],
            "available_fp": stats["available"]["fp"],
            "available_fn": stats["available"]["fn"],
            "occupied_tp":  stats["occupied"]["tp"],
            "occupied_fp":  stats["occupied"]["fp"],
            "occupied_fn":  stats["occupied"]["fn"],
        })
        
        if i % 10 == 0 or i == len(images):
            print(f"   Processed {i}/{len(images)}")
    
    # Calculate final metrics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    final_metrics = {}
    for cls in total_stats:
        tp = total_stats[cls]["tp"]
        fp = total_stats[cls]["fp"]
        fn = total_stats[cls]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        final_metrics[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # Print results table
    print(f"\n   {'Class':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<8}")
    print(f"   {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
    
    for cls, m in final_metrics.items():
        print(f"   {cls:<12} {m['tp']:<6} {m['fp']:<6} {m['fn']:<6} "
              f"{m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<8.4f}")
    
    # Overall (macro average)
    avg_precision = sum(m["precision"] for m in final_metrics.values()) / len(final_metrics)
    avg_recall    = sum(m["recall"] for m in final_metrics.values()) / len(final_metrics)
    avg_f1        = sum(m["f1"] for m in final_metrics.values()) / len(final_metrics)
    
    print(f"   {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
    print(f"   {'AVERAGE':<12} {'':<6} {'':<6} {'':<6} "
          f"{avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<8.4f}")
    
    # Save CSV
    csv_path = "overlap_evaluation_per_image.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_image_results[0].keys())
        writer.writeheader()
        writer.writerows(per_image_results)
    print(f"\n✅ Per-image results: {csv_path}")
    
    # Save summary
    summary_path = "overlap_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  OVERLAP METHOD EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model:               {MODEL} (COCO pretrained, no custom training)\n")
        f.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"Overlap threshold:    {OVERLAP_THRESHOLD}\n")
        f.write(f"Test images:          {len(images)}\n\n")
        
        f.write(f"{'Class':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<8}\n")
        f.write("-" * 70 + "\n")
        for cls, m in final_metrics.items():
            f.write(f"{cls:<12} {m['tp']:<6} {m['fp']:<6} {m['fn']:<6} "
                   f"{m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<8.4f}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'AVERAGE':<12} {'':<6} {'':<6} {'':<6} "
               f"{avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<8.4f}\n\n")
        
        f.write("\nCOMPARISON TO TRAINED MODEL (exp4 - YOLOv8m weighted)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Trained model:  Precision=0.630  Recall=0.427  mAP@0.5=0.458\n")
        f.write(f"Overlap method: Precision={avg_precision:.4f}  Recall={avg_recall:.4f}  F1={avg_f1:.4f}\n")
    
    print(f"✅ Summary saved:    {summary_path}\n")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    evaluate_dataset()