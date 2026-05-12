"""
==============================================================
   SINGLE EXPERIMENT - WEIGHTED LOSS FOR OCCUPIED CLASS
   Model: YOLOv8m @ 1280
   Goal: Boost occupied class performance
==============================================================

WHAT THIS DOES:
  Trains one YOLOv8m model at 1280 resolution with increased
  classification loss weight. This tells the model to pay
  extra attention to getting class labels right, which can
  help the weaker 'occupied' class.

EXPECTED TIME: ~1.5 - 2 hours on RTX 3060
EXPECTED IMPROVEMENT: Occupied mAP 0.288 → 0.35+

HOW TO RUN:
  python train_weighted.py
"""

import torch
from roboflow import Roboflow
from ultralytics import YOLO
import os
import sys
import time
from datetime import datetime


# ==============================================================
#  CONFIGURATION - SET YOUR API KEY
# ==============================================================

ROBOFLOW_API_KEY = "DFRAPJ5ad6Ubpu6wZEj3"
WORKSPACE        = "kareems-workspace-1tyaf"
PROJECT          = "seats-detection-lfkct"
VERSION_NUMBER   = 4

# Training settings
MODEL_VARIANT = "yolov8m.pt"
EPOCHS        = 80
IMAGE_SIZE    = 1280
BATCH_SIZE    = 4
PATIENCE      = 20
CLS_WEIGHT    = 1.5          # KEY CHANGE: classification loss weight (default is 0.5)
RUN_NAME      = "exp4_weighted_occupied"
SEED          = 42


# ==============================================================
#  GPU CHECK
# ==============================================================

def check_gpu():
    print("=" * 70)
    print("  GPU CHECK")
    print("=" * 70)
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        sys.exit(1)
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


# ==============================================================
#  CHECK DATASET (no need to re-download if already exists)
# ==============================================================

def get_dataset():
    print("=" * 70)
    print("  DATASET")
    print("=" * 70)

    # Check if dataset already exists locally
    existing_path = "Seats-Detection-4/data.yaml"
    if os.path.exists(existing_path):
        print(f"\n✅ Using existing dataset at: {existing_path}\n")
        return existing_path

    # Otherwise download fresh
    if ROBOFLOW_API_KEY == "YOUR_NEW_API_KEY_HERE":
        print("\n❌ Dataset not found and API key not set!")
        sys.exit(1)

    print("\n   Downloading from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION_NUMBER)
    dataset = version.download("yolov8")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"\n✅ Dataset ready: {data_yaml}\n")
    return data_yaml


# ==============================================================
#  TRAIN WITH WEIGHTED LOSS
# ==============================================================

def train_weighted(data_yaml):
    print("=" * 70)
    print(f"  TRAINING: {RUN_NAME}")
    print("=" * 70)
    print(f"   Model:        {MODEL_VARIANT}")
    print(f"   Image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Batch size:   {BATCH_SIZE}")
    print(f"   Epochs:       {EPOCHS}")
    print(f"   CLS weight:   {CLS_WEIGHT} (default is 0.5)")
    print(f"   Started:      {datetime.now().strftime('%H:%M:%S')}")
    print(f"\n   Expected time: 1.5 - 2 hours\n")

    start_time = time.time()

    model = YOLO(MODEL_VARIANT)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        cls=CLS_WEIGHT,           # weighted classification loss
        name=RUN_NAME,
        seed=SEED,
        device=0,
        workers=4,
        save=True,
        verbose=True,
        plots=True,
    )

    duration = (time.time() - start_time) / 60
    print(f"\n✅ Training complete in {duration:.1f} minutes")
    return model


# ==============================================================
#  VALIDATE WITH PER-CLASS BREAKDOWN
# ==============================================================

def validate_full(model, data_yaml):
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    # Default threshold validation
    print("\n--- Default threshold (conf=0.25) ---")
    metrics = model.val(data=data_yaml, verbose=True)

    # Test multiple thresholds
    print("\n--- Testing multiple thresholds ---")
    for conf in [0.15, 0.20, 0.25, 0.30]:
        m = model.val(data=data_yaml, conf=conf, verbose=False)
        print(f"   conf={conf}: mAP@0.5={m.box.map50:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}")


# ==============================================================
#  SUMMARY
# ==============================================================

def print_summary():
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n   Results saved to: runs/detect/{RUN_NAME}/")
    print(f"\n   Important files:")
    print(f"   • Best model:        runs/detect/{RUN_NAME}/weights/best.pt")
    print(f"   • Confusion matrix:  runs/detect/{RUN_NAME}/confusion_matrix.png")
    print(f"   • PR curve:          runs/detect/{RUN_NAME}/BoxPR_curve.png")
    print(f"   • Results:           runs/detect/{RUN_NAME}/results.png")
    print("=" * 70 + "\n")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print("  WEIGHTED LOSS EXPERIMENT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯" * 35 + "\n")

    overall_start = time.time()

    check_gpu()
    data_yaml = get_dataset()
    model = train_weighted(data_yaml)
    validate_full(model, data_yaml)
    print_summary()

    total_time = (time.time() - overall_start) / 60
    print(f"\n   Total time: {total_time:.1f} minutes\n")