"""
==============================================================
   FINAL OPTIMIZED TRAINING - LAST ATTEMPT
   YOLOv8m + 1280 + Weighted Loss + AdamW + Cosine LR + TTA
==============================================================

WHAT THIS DOES:
  Combines every successful optimization from previous experiments:
  - YOLOv8m (best balance from your tests)
  - 1280x1280 resolution (resolution helped most)
  - Weighted classification loss (helped occupied class)
  - AdamW optimizer (often beats SGD for fine-tuning)
  - Cosine learning rate schedule (smoother training)
  - 150 epochs with patience 30 (more training time)
  - Test Time Augmentation in validation (free accuracy boost)

EXPECTED TIME: ~3 hours on RTX 3060
EXPECTED RESULT: mAP 0.46-0.50 (small but real improvement)

HOW TO RUN:
  python train_final.py
"""

import torch
from roboflow import Roboflow
from ultralytics import YOLO
import os
import sys
import time
from datetime import datetime


# ==============================================================
#  CONFIGURATION
# ==============================================================

ROBOFLOW_API_KEY = "YOUR_NEW_API_KEY_HERE"
WORKSPACE        = "kareems-workspace-1tyaf"
PROJECT          = "seats-detection-lfkct"
VERSION_NUMBER   = 4

# OPTIMAL TRAINING SETTINGS
MODEL_VARIANT    = "yolov8m.pt"
EPOCHS           = 150              # more epochs with early stopping
IMAGE_SIZE       = 1280             # high resolution
BATCH_SIZE       = 4                # fits in 12GB VRAM at 1280
PATIENCE         = 30               # wait longer before early stop
RUN_NAME         = "exp5_final_optimized"
SEED             = 42

# ADVANCED OPTIMIZATIONS
OPTIMIZER        = "AdamW"          # often better than SGD for fine-tuning
LR_INITIAL       = 0.001            # AdamW friendly learning rate
COSINE_LR        = True             # smoother learning rate decay
CLS_WEIGHT       = 1.5              # weighted loss for class imbalance
BOX_WEIGHT       = 7.5              # default box weight
DFL_WEIGHT       = 1.5              # default DFL weight
WARMUP_EPOCHS    = 5                # gradual warmup
MOMENTUM         = 0.937
WEIGHT_DECAY     = 0.0005           # regularization to prevent overfitting

# DATA AUGMENTATION (during training)
HSV_H            = 0.015            # hue augmentation
HSV_S            = 0.7              # saturation augmentation
HSV_V            = 0.4              # value/brightness augmentation
FLIPLR           = 0.5              # horizontal flip probability
MOSAIC           = 1.0              # mosaic augmentation
MIXUP            = 0.1              # mixup augmentation


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
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   PyTorch: {torch.__version__}\n")


# ==============================================================
#  CHECK DATASET
# ==============================================================

def get_dataset():
    print("=" * 70)
    print("  DATASET")
    print("=" * 70)

    existing_path = "Seats-Detection-4/data.yaml"
    if os.path.exists(existing_path):
        print(f"\n✅ Using existing dataset: {existing_path}\n")
        return existing_path

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
#  TRAIN WITH ALL OPTIMIZATIONS
# ==============================================================

def train_optimized(data_yaml):
    print("=" * 70)
    print(f"  TRAINING: {RUN_NAME}")
    print("=" * 70)
    print(f"   Model:           {MODEL_VARIANT}")
    print(f"   Resolution:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Batch:           {BATCH_SIZE}")
    print(f"   Epochs:          {EPOCHS} (patience {PATIENCE})")
    print(f"   Optimizer:       {OPTIMIZER}")
    print(f"   Initial LR:      {LR_INITIAL}")
    print(f"   Cosine LR:       {COSINE_LR}")
    print(f"   CLS weight:      {CLS_WEIGHT}")
    print(f"   Weight decay:    {WEIGHT_DECAY}")
    print(f"   Warmup epochs:   {WARMUP_EPOCHS}")
    print(f"\n   Augmentations:")
    print(f"   - HSV (H/S/V):   {HSV_H}/{HSV_S}/{HSV_V}")
    print(f"   - Flip LR:       {FLIPLR}")
    print(f"   - Mosaic:        {MOSAIC}")
    print(f"   - MixUp:         {MIXUP}")
    print(f"\n   Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Expected: ~3 hours\n")

    start_time = time.time()

    model = YOLO(MODEL_VARIANT)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        # Optimizer settings
        optimizer=OPTIMIZER,
        lr0=LR_INITIAL,
        lrf=0.01,                 # final LR fraction
        cos_lr=COSINE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        # Loss weights
        cls=CLS_WEIGHT,
        box=BOX_WEIGHT,
        dfl=DFL_WEIGHT,
        # Augmentations
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        # Other
        name=RUN_NAME,
        seed=SEED,
        device=0,
        workers=4,
        save=True,
        verbose=True,
        plots=True,
        amp=True,                 # automatic mixed precision = faster
    )

    duration = (time.time() - start_time) / 60
    print(f"\n✅ Training complete in {duration:.1f} minutes ({duration/60:.1f} hours)")
    return model


# ==============================================================
#  VALIDATION WITH TTA
# ==============================================================

def validate_full(model, data_yaml):
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    # Standard validation
    print("\n--- Standard validation (no TTA) ---")
    m_std = model.val(data=data_yaml, verbose=True)

    # TTA validation (Test Time Augmentation)
    print("\n--- TTA validation (Test Time Augmentation) ---")
    m_tta = model.val(data=data_yaml, augment=True, verbose=True)

    # Threshold sweep
    print("\n--- Threshold sweep (with TTA) ---")
    best_thresh = 0.25
    best_map = 0
    for conf in [0.15, 0.20, 0.25, 0.30, 0.35]:
        m = model.val(data=data_yaml, conf=conf, augment=True, verbose=False)
        marker = ""
        if m.box.map50 > best_map:
            best_map = m.box.map50
            best_thresh = conf
            marker = " ← BEST"
        print(f"   conf={conf}: mAP@0.5={m.box.map50:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}{marker}")

    print(f"\n   Best threshold: {best_thresh} with mAP={best_map:.4f}")


# ==============================================================
#  PRINT SUMMARY
# ==============================================================

def print_summary():
    print("\n" + "=" * 70)
    print("  FINAL OPTIMIZED MODEL - DONE")
    print("=" * 70)
    print(f"\n   Results saved to: runs/detect/{RUN_NAME}/")
    print(f"\n   Files for report:")
    print(f"   • Best model:        runs/detect/{RUN_NAME}/weights/best.pt")
    print(f"   • Confusion matrix:  runs/detect/{RUN_NAME}/confusion_matrix.png")
    print(f"   • PR curve:          runs/detect/{RUN_NAME}/BoxPR_curve.png")
    print(f"   • Training curves:   runs/detect/{RUN_NAME}/results.png")
    print(f"\n   For demo: use augment=True in predict for best accuracy")
    print("=" * 70 + "\n")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("  FINAL OPTIMIZED TRAINING - LAST ATTEMPT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀" * 35 + "\n")

    overall_start = time.time()

    check_gpu()
    data_yaml = get_dataset()
    model = train_optimized(data_yaml)
    validate_full(model, data_yaml)
    print_summary()

    total_time = (time.time() - overall_start) / 60
    print(f"\n   Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)\n")
