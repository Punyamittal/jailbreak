"""Quick test to verify ML training setup."""

import sys

print("Checking dependencies...")

try:
    import sklearn
    print(f"[OK] scikit-learn {sklearn.__version__}")
except ImportError:
    print("[ERROR] scikit-learn not installed")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import numpy
    print(f"[OK] numpy {numpy.__version__}")
except ImportError:
    print("[ERROR] numpy not installed")
    print("Install with: pip install numpy")
    sys.exit(1)

try:
    from train_ml_model import AntiJailbreakMLModel
    print("[OK] train_ml_model module imported")
except ImportError as e:
    print(f"[ERROR] Failed to import train_ml_model: {e}")
    sys.exit(1)

# Check if dataset exists
from pathlib import Path
dataset_path = Path("datasets/malignant_labeled.jsonl")
if dataset_path.exists():
    print(f"[OK] Dataset found: {dataset_path}")
else:
    print(f"[ERROR] Dataset not found: {dataset_path}")
    print("Run: python process_malignant_dataset.py first")
    sys.exit(1)

print("\n[SUCCESS] All checks passed!")
print("\nYou can now run: python train_ml_model.py")

