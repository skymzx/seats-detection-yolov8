"""
==============================================================
   LATENCY BENCHMARK
   Measures real-world inference speed of your final model
==============================================================

WHAT THIS DOES:
  1. Loads your best model
  2. Runs inference on test images
  3. Measures preprocessing, inference, and postprocessing time
  4. Tests on both GPU and CPU
  5. Reports FPS (frames per second)
  6. Saves results to a text file for the report

HOW TO RUN:
  python benchmark_latency.py
"""

import torch
from ultralytics import YOLO
import time
import os
import platform
import sys


# ==============================================================
#  CONFIGURATION
# ==============================================================

MODEL_PATH = "runs/detect/exp4_weighted_occupied/weights/best.pt"
TEST_FOLDER = "Seats-Detection-4/test/images"
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


# ==============================================================
#  SYSTEM INFO
# ==============================================================

def get_system_info():
    print("=" * 70)
    print("  SYSTEM SPECIFICATIONS")
    print("=" * 70)
    
    info = {}
    
    # CPU
    info["CPU"] = platform.processor() or "Unknown"
    info["OS"] = f"{platform.system()} {platform.release()}"
    info["Python"] = platform.python_version()
    info["PyTorch"] = torch.__version__
    
    # GPU
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["CUDA"] = torch.version.cuda
    else:
        info["GPU"] = "Not available"
    
    for key, value in info.items():
        print(f"   {key:12s}: {value}")
    print()
    
    return info


# ==============================================================
#  GET TEST IMAGES
# ==============================================================

def get_test_images():
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ Test folder not found: {TEST_FOLDER}")
        print("   Trying alternative paths...")
        
        for alt in ["test/images", "valid/images", "Seats-Detection-4/valid/images"]:
            if os.path.exists(alt):
                print(f"   Found: {alt}")
                return [os.path.join(alt, f) for f in os.listdir(alt) if f.endswith(('.jpg', '.png'))]
        
        print("\n❌ No test images found. Please check paths.")
        sys.exit(1)
    
    images = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
              if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(images)} test images")
    return images


# ==============================================================
#  BENCHMARK ON DEVICE
# ==============================================================

def benchmark_device(model_path, images, device, device_name):
    print("\n" + "=" * 70)
    print(f"  BENCHMARKING ON {device_name}")
    print("=" * 70)
    
    model = YOLO(model_path)
    
    # Warmup runs (first inference is always slower)
    print(f"\n   Warming up ({WARMUP_RUNS} runs)...")
    for i in range(WARMUP_RUNS):
        _ = model.predict(images[i % len(images)], device=device, verbose=False)
    
    # Actual benchmark
    print(f"   Benchmarking ({BENCHMARK_RUNS} runs)...")
    
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    
    for i in range(BENCHMARK_RUNS):
        img = images[i % len(images)]
        results = model.predict(img, device=device, verbose=False)
        
        # Get timing from YOLOv8 results
        speed = results[0].speed
        preprocess_times.append(speed['preprocess'])
        inference_times.append(speed['inference'])
        postprocess_times.append(speed['postprocess'])
    
    # Calculate averages
    avg_pre = sum(preprocess_times) / len(preprocess_times)
    avg_inf = sum(inference_times) / len(inference_times)
    avg_post = sum(postprocess_times) / len(postprocess_times)
    avg_total = avg_pre + avg_inf + avg_post
    fps = 1000 / avg_total if avg_total > 0 else 0
    
    print(f"\n   RESULTS ({device_name}):")
    print(f"   Preprocess:    {avg_pre:.2f} ms")
    print(f"   Inference:     {avg_inf:.2f} ms")
    print(f"   Postprocess:   {avg_post:.2f} ms")
    print(f"   Total/image:   {avg_total:.2f} ms")
    print(f"   FPS:           {fps:.1f} frames/second")
    
    return {
        "device": device_name,
        "preprocess_ms": avg_pre,
        "inference_ms": avg_inf,
        "postprocess_ms": avg_post,
        "total_ms": avg_total,
        "fps": fps
    }


# ==============================================================
#  SAVE REPORT
# ==============================================================

def save_report(system_info, results):
    output_file = "latency_benchmark_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  LATENCY BENCHMARK REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SYSTEM SPECIFICATIONS\n")
        f.write("-" * 70 + "\n")
        for key, value in system_info.items():
            f.write(f"   {key:12s}: {value}\n")
        f.write("\n")
        
        f.write("MODEL\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Model:        YOLOv8m (weighted loss)\n")
        f.write(f"   Input size:   1280 x 1280\n")
        f.write(f"   Weights:      {MODEL_PATH}\n\n")
        
        f.write("BENCHMARK RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Device':<15} {'Pre (ms)':<12} {'Inf (ms)':<12} {'Post (ms)':<12} {'Total (ms)':<12} {'FPS':<8}\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            f.write(f"{r['device']:<15} {r['preprocess_ms']:<12.2f} {r['inference_ms']:<12.2f} "
                   f"{r['postprocess_ms']:<12.2f} {r['total_ms']:<12.2f} {r['fps']:<8.1f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("NOTES\n")
        f.write("-" * 70 + "\n")
        f.write("- Preprocess: image resize, normalization, tensor conversion\n")
        f.write("- Inference:  forward pass through the neural network\n")
        f.write("- Postprocess: non-maximum suppression, box filtering\n")
        f.write("- Total: complete time per image\n")
        f.write("- FPS: frames per second the model can process\n\n")
        f.write("- Real-time threshold: typically 30 FPS (33.3 ms total)\n")
        f.write("- Surveillance suitable: 5+ FPS (200 ms total)\n")
    
    print(f"\n\n✅ Results saved to: {output_file}")
    return output_file


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("\n" + "📊" * 35)
    print("  LATENCY BENCHMARK")
    print("📊" * 35 + "\n")
    
    system_info = get_system_info()
    images = get_test_images()
    
    if len(images) == 0:
        print("❌ No images to benchmark")
        sys.exit(1)
    
    results = []
    
    # Benchmark on GPU
    if torch.cuda.is_available():
        gpu_result = benchmark_device(MODEL_PATH, images, 0, f"GPU ({system_info['GPU']})")
        results.append(gpu_result)
    
    # Benchmark on CPU
    cpu_result = benchmark_device(MODEL_PATH, images, "cpu", "CPU")
    results.append(cpu_result)
    
    # Save report
    output_file = save_report(system_info, results)
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"   {r['device']:<30} {r['total_ms']:.2f} ms/image  ({r['fps']:.1f} FPS)")
    print("=" * 70 + "\n")
