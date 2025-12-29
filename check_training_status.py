"""
Quick script to check training status and show progress.
"""

from pathlib import Path
import json

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)

# Check if relabeled dataset exists
relabeled_file = Path("datasets/relabeled/all_relabeled_combined.jsonl")
if relabeled_file.exists():
    count = sum(1 for _ in open(relabeled_file, 'r', encoding='utf-8'))
    print(f"\n[OK] Relabeled dataset exists: {count:,} examples")
else:
    print("\n[ERROR] Relabeled dataset not found!")
    print("Run: python relabel_dataset.py")

# Check if model directory exists
model_dir = Path("models/security")
if model_dir.exists():
    print(f"\n[OK] Model directory exists")
    
    model_files = {
        "security_model.pkl": "Main model",
        "security_vectorizer.pkl": "Vectorizer",
        "security_encoder.pkl": "Label encoder",
        "security_threshold.txt": "Threshold"
    }
    
    for file, desc in model_files.items():
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {file} ({desc}): {size_mb:.2f} MB")
        else:
            print(f"  [MISSING] {file} ({desc})")
else:
    print(f"\n[WAITING] Model directory does not exist")
    print("Training may still be running or hasn't started")

# Check for any error logs
error_log = Path("training_log.txt")
if error_log.exists():
    print(f"\n[INFO] Training log found")
    with open(error_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines:
            print(f"  Last 5 lines:")
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")

print("\n" + "="*70)
print("To train the model, run:")
print("  python train_security_model.py")
print("="*70)

