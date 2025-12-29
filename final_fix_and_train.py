"""
Final fix and training script.
Clears caches, verifies dependencies, and trains the model.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

print("="*70)
print("FINAL FIX AND TRAIN")
print("="*70)

# Step 1: Clear Python cache
print("\n[1/5] Clearing Python cache...")
for root, dirs, files in os.walk("."):
    if "__pycache__" in dirs:
        cache_dir = os.path.join(root, "__pycache__")
        try:
            shutil.rmtree(cache_dir)
            print(f"  Removed: {cache_dir}")
        except:
            pass
    for file in files:
        if file.endswith(".pyc"):
            try:
                os.remove(os.path.join(root, file))
            except:
                pass
print("  [OK] Cache cleared")

# Step 2: Verify and fix NumPy
print("\n[2/5] Verifying NumPy...")
try:
    import numpy
    if numpy.__version__.startswith('2.'):
        print(f"  NumPy {numpy.__version__} detected - downgrading...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2", "--force-reinstall"], check=True)
        import importlib
        importlib.reload(numpy)
    print(f"  [OK] NumPy {numpy.__version__}")
except Exception as e:
    print(f"  [ERROR] NumPy issue: {e}")
    sys.exit(1)

# Step 3: Verify tf-keras
print("\n[3/5] Verifying tf-keras...")
try:
    import tf_keras
    print("  [OK] tf-keras installed")
except ImportError:
    print("  Installing tf-keras...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tf-keras"], check=True)
    import tf_keras
    print("  [OK] tf-keras installed")

# Step 4: Test imports
print("\n[4/5] Testing imports...")
try:
    from typing import Optional, List, Tuple
    print("  [OK] typing imports")
    
    from intent_based_detector import IntentBasedDetector
    print("  [OK] IntentBasedDetector import")
    
    from train_intent_model import train_intent_model
    print("  [OK] train_intent_model import")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Check dataset
print("\n[5/5] Checking dataset...")
dataset_paths = [
    "datasets/combined_with_custom.jsonl",
    "datasets/combined_with_hf.jsonl",
    "datasets/combined_training_dataset.jsonl"
]

dataset_found = None
for path in dataset_paths:
    if Path(path).exists():
        dataset_found = path
        print(f"  [OK] Found: {path}")
        break

if not dataset_found:
    print("  [ERROR] No dataset found!")
    sys.exit(1)

print("\n" + "="*70)
print("[SUCCESS] ALL CHECKS PASSED!")
print("="*70)
print("\nStarting training...")
print("="*70)

# Now run training
try:
    from train_intent_model import train_intent_model
    detector = train_intent_model(
        dataset_path=dataset_found,
        jailbreak_threshold=0.4
    )
    
    if detector:
        print("\n" + "="*70)
        print("[SUCCESS] TRAINING COMPLETE!")
        print("="*70)
    else:
        print("\n[ERROR] Training returned None")
        sys.exit(1)
        
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

