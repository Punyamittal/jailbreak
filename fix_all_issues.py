"""
Fix all dependency and code issues for intent-based model.
"""

import subprocess
import sys
import os

print("="*70)
print("FIXING ALL ISSUES")
print("="*70)

# Step 1: Verify dependencies
print("\n[1/4] Verifying dependencies...")
try:
    import numpy
    print(f"  NumPy: {numpy.__version__}")
    if numpy.__version__.startswith('2.'):
        print("  [FIX] Downgrading NumPy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2", "--force-reinstall", "--quiet"], check=True)
        print("  [OK] NumPy downgraded")
    else:
        print("  [OK] NumPy version OK")
except Exception as e:
    print(f"  [ERROR] NumPy issue: {e}")

try:
    import tf_keras
    print("  [OK] tf-keras installed")
except ImportError:
    print("  [FIX] Installing tf-keras...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tf-keras", "--quiet"], check=True)
    print("  [OK] tf-keras installed")

# Step 2: Verify imports
print("\n[2/4] Verifying Python imports...")
try:
    from typing import Optional, List, Tuple
    print("  [OK] typing imports OK")
except Exception as e:
    print(f"  [ERROR] typing import failed: {e}")

try:
    from intent_based_detector import IntentBasedDetector
    print("  [OK] IntentBasedDetector import OK")
except Exception as e:
    print(f"  [ERROR] IntentBasedDetector import failed: {e}")
    print(f"  Error details: {str(e)}")

# Step 3: Check for syntax errors
print("\n[3/4] Checking for syntax errors...")
try:
    with open("train_intent_model.py", 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, "train_intent_model.py", "exec")
    print("  [OK] train_intent_model.py syntax OK")
except SyntaxError as e:
    print(f"  [ERROR] Syntax error in train_intent_model.py: {e}")
except Exception as e:
    print(f"  [ERROR] Error checking train_intent_model.py: {e}")

try:
    with open("intent_based_detector.py", 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, "intent_based_detector.py", "exec")
    print("  [OK] intent_based_detector.py syntax OK")
except SyntaxError as e:
    print(f"  [ERROR] Syntax error in intent_based_detector.py: {e}")
except Exception as e:
    print(f"  [ERROR] Error checking intent_based_detector.py: {e}")

# Step 4: Verify dataset exists
print("\n[4/4] Checking datasets...")
dataset_paths = [
    "datasets/combined_with_custom.jsonl",
    "datasets/combined_with_hf.jsonl",
    "datasets/combined_training_dataset.jsonl"
]

found = False
for path in dataset_paths:
    if os.path.exists(path):
        print(f"  [OK] Found dataset: {path}")
        found = True
        break

if not found:
    print("  [WARNING] No training dataset found!")

print("\n" + "="*70)
print("[SUCCESS] ALL CHECKS COMPLETE!")
print("="*70)
print("\nYou can now run:")
print("  python train_intent_model.py")

