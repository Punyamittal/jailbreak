"""
Complete workflow: Add custom data and retrain model.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Add custom data and retrain."""
    print("="*70)
    print("ADDING CUSTOM DATASET & RETRAINING")
    print("="*70)
    
    # Step 1: Add custom data
    print("\n[STEP 1/2] Adding custom dataset to training...")
    try:
        result = subprocess.run(
            [sys.executable, "add_custom_to_training.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("[OK] Custom data added")
            if result.stdout:
                print(result.stdout[-300:])
        else:
            print(f"[WARNING] {result.stderr}")
    except Exception as e:
        print(f"[WARNING] {e}")
    
    # Step 2: Retrain
    print("\n[STEP 2/2] Retraining model with custom data...")
    print("This may take 10-30 minutes...")
    try:
        result = subprocess.run(
            [sys.executable, "train_balanced_model.py"],
            timeout=1800  # 30 minutes
        )
        if result.returncode == 0:
            print("\n[SUCCESS] Model retrained!")
        else:
            print("\n[WARNING] Training had issues")
    except subprocess.TimeoutExpired:
        print("[WARNING] Training taking longer than expected")
    except Exception as e:
        print(f"[WARNING] {e}")
    
    print("\n" + "="*70)
    print("[SUCCESS] COMPLETE!")
    print("="*70)
    print("\nNext: Test on custom dataset:")
    print("  python test_custom_dataset.py")

if __name__ == "__main__":
    main()

