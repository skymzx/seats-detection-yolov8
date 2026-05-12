"""
==============================================================
   OVERNIGHT TRAINING - MULTIPLE EXPERIMENTS
   Runs 3 model variants back-to-back
   Tests each at 4 different confidence thresholds
==============================================================

WHAT THIS SCRIPT DOES:
  1. Trains YOLOv8m at 640x640
  2. Trains YOLOv8m at 1280x1280
  3. Trains YOLOv8l at 640x640
  4. Evaluates each model at conf thresholds: 0.20, 0.25, 0.30, 0.40
  5. Saves all results in organized folders
  6. Generates a SUMMARY.txt comparing everything

EXPECTED TIME: 4-5 hours on RTX 3060
SLEEP WELL.

HOW TO RUN:
  python overnight_train.py
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

# All experiments to run
EXPERIMENTS = [
    {
        "name":  "exp1_yolov8m_640",
        "model": "yolov8m.pt",
        "imgsz": 640,
        "batch": 8,
        "epochs": 100,
    },
    {
        "name":  "exp2_yolov8m_1280",
        "model": "yolov8m.pt",
        "imgsz": 1280,
        "batch": 4,           # smaller batch needed for bigger images
        "epochs": 80,         # slightly fewer epochs since each takes longer
    },
    {
        "name":  "exp3_yolov8l_640",
        "model": "yolov8l.pt",
        "imgsz": 640,
        "batch": 4,           # larger model needs smaller batch
        "epochs": 100,
    },
]

# Confidence thresholds to test each model at
THRESHOLDS = [0.20, 0.25, 0.30, 0.40]


# ==============================================================
#  GPU CHECK
# ==============================================================

def check_gpu():
    print("=" * 70)
    print("  GPU CHECK")
    print("=" * 70)
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Fix this before running overnight training.")
        sys.exit(1)
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   PyTorch: {torch.__version__}\n")


# ==============================================================
#  DOWNLOAD DATASET (only once)
# ==============================================================

def download_dataset():
    print("=" * 70)
    print("  DOWNLOADING DATASET")
    print("=" * 70)

    if ROBOFLOW_API_KEY == "YOUR_NEW_API_KEY_HERE":
        print("\n❌ Set your Roboflow API key at the top of this script!")
        sys.exit(1)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION_NUMBER)
    dataset = version.download("yolov8")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"\n✅ Dataset ready: {data_yaml}\n")
    return data_yaml


# ==============================================================
#  RUN ONE EXPERIMENT
# ==============================================================

def run_experiment(exp, data_yaml):
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {exp['name']}")
    print("=" * 70)
    print(f"   Model:     {exp['model']}")
    print(f"   Image:     {exp['imgsz']}x{exp['imgsz']}")
    print(f"   Batch:     {exp['batch']}")
    print(f"   Epochs:    {exp['epochs']}")
    print(f"   Started:   {datetime.now().strftime('%H:%M:%S')}\n")

    start_time = time.time()

    try:
        model = YOLO(exp['model'])

        results = model.train(
            data=data_yaml,
            epochs=exp['epochs'],
            imgsz=exp['imgsz'],
            batch=exp['batch'],
            patience=25,
            name=exp['name'],
            seed=42,
            device=0,
            workers=4,
            save=True,
            verbose=False,    # less output during overnight
            plots=True,
        )

        duration = (time.time() - start_time) / 60
        print(f"\n✅ Training done in {duration:.1f} minutes")

        # Evaluate at different thresholds
        eval_at_thresholds(model, exp['name'])

        return {
            "name": exp['name'],
            "success": True,
            "duration": duration,
            "metrics": evaluate_default(model)
        }

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        return {
            "name": exp['name'],
            "success": False,
            "error": str(e)
        }


# ==============================================================
#  EVALUATE AT DEFAULT THRESHOLD
# ==============================================================

def evaluate_default(model):
    """Standard validation at default threshold"""
    metrics = model.val(verbose=False)
    return {
        "mAP50":    float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":   float(metrics.box.mr),
    }


# ==============================================================
#  EVALUATE AT MULTIPLE THRESHOLDS
# ==============================================================

def eval_at_thresholds(model, exp_name):
    """Run validation at multiple confidence thresholds and save results"""
    print(f"\n   Testing at multiple confidence thresholds...")

    threshold_dir = f"runs/detect/{exp_name}/thresholds"
    os.makedirs(threshold_dir, exist_ok=True)

    threshold_results = {}

    for conf in THRESHOLDS:
        try:
            metrics = model.val(conf=conf, verbose=False)
            result = {
                "conf":      conf,
                "mAP50":     float(metrics.box.map50),
                "mAP50-95":  float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall":    float(metrics.box.mr),
            }
            threshold_results[conf] = result

            # Save individual threshold result
            result_file = f"{threshold_dir}/conf_{conf:.2f}_results.txt"
            with open(result_file, 'w') as f:
                f.write(f"Confidence Threshold: {conf}\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"mAP@0.5:      {result['mAP50']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {result['mAP50-95']:.4f}\n")
                f.write(f"Precision:    {result['precision']:.4f}\n")
                f.write(f"Recall:       {result['recall']:.4f}\n")

            print(f"   conf={conf}: mAP={result['mAP50']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}")

        except Exception as e:
            print(f"   conf={conf}: FAILED ({e})")

    return threshold_results


# ==============================================================
#  GENERATE SUMMARY
# ==============================================================

def generate_summary(all_results):
    print("\n" + "=" * 70)
    print("  GENERATING SUMMARY")
    print("=" * 70)

    summary_path = "runs/SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  OVERNIGHT EXPERIMENT RESULTS SUMMARY\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Overall comparison table
        f.write("EXPERIMENT COMPARISON (at default conf=0.25)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Experiment':<25} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-" * 70 + "\n")

        for r in all_results:
            if r.get('success'):
                m = r['metrics']
                f.write(f"{r['name']:<25} {m['mAP50']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f}\n")
            else:
                f.write(f"{r['name']:<25} FAILED: {r.get('error', 'unknown')[:40]}\n")

        f.write("\n" + "=" * 70 + "\n\n")

        # Detailed per-experiment threshold breakdown
        for r in all_results:
            if not r.get('success'):
                continue

            f.write(f"\n{r['name']} - THRESHOLD ANALYSIS\n")
            f.write("-" * 70 + "\n")

            threshold_dir = f"runs/detect/{r['name']}/thresholds"
            if os.path.exists(threshold_dir):
                for conf in THRESHOLDS:
                    result_file = f"{threshold_dir}/conf_{conf:.2f}_results.txt"
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as rf:
                            f.write(rf.read())
                            f.write("\n")

            f.write(f"Training time: {r['duration']:.1f} minutes\n")

        # Recommendations
        f.write("\n" + "=" * 70 + "\n")
        f.write("  WHICH ONE TO USE\n")
        f.write("=" * 70 + "\n\n")

        successful = [r for r in all_results if r.get('success')]
        if successful:
            best = max(successful, key=lambda x: x['metrics']['mAP50'])
            f.write(f"Best overall mAP: {best['name']} (mAP@0.5 = {best['metrics']['mAP50']:.4f})\n")
            f.write(f"Best model file:  runs/detect/{best['name']}/weights/best.pt\n\n")
            f.write("FILES TO USE FOR YOUR REPORT:\n")
            f.write(f"  runs/detect/{best['name']}/weights/best.pt\n")
            f.write(f"  runs/detect/{best['name']}/confusion_matrix.png\n")
            f.write(f"  runs/detect/{best['name']}/results.png\n")
            f.write(f"  runs/detect/{best['name']}/BoxPR_curve.png\n")

    print(f"\n✅ Summary saved to: {summary_path}\n")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("\n" + "🌙" * 35)
    print("  OVERNIGHT TRAINING SESSION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Experiments queued: {len(EXPERIMENTS)}")
    print("🌙" * 35 + "\n")

    overall_start = time.time()

    check_gpu()
    data_yaml = download_dataset()

    all_results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> Running experiment {i}/{len(EXPERIMENTS)}")
        result = run_experiment(exp, data_yaml)
        all_results.append(result)

    generate_summary(all_results)

    total_time = (time.time() - overall_start) / 60
    print("\n" + "☀️" * 35)
    print("  ALL DONE - GOOD MORNING!")
    print(f"  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"  Check runs/SUMMARY.txt for results")
    print("☀️" * 35 + "\n")
