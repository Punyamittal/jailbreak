"""
Complete workflow: Process datasets and train model.
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    """Process datasets and train model."""
    print("="*70)
    print("COMPLETE WORKFLOW: PROCESS & TRAIN")
    print("="*70)
    
    # Step 1: Process datasets
    print("\n[STEP 1] Processing large datasets...")
    print("This may take 5-15 minutes for large files...")
    
    try:
        result = subprocess.run(
            [sys.executable, "process_large_datasets.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes max
        )
        
        if result.returncode == 0:
            print("[OK] Dataset processing complete!")
            if result.stdout:
                # Show last 20 lines of output
                lines = result.stdout.strip().split('\n')
                print("\n".join(lines[-20:]))
        else:
            print("[ERROR] Processing failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("[WARNING] Processing is taking longer than expected.")
        print("It may still be running in the background.")
        print("Check status with: python check_processing_status.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False
    
    # Step 2: Check if combined dataset exists
    combined_path = Path("datasets/combined_training_dataset.jsonl")
    if not combined_path.exists():
        print("\n[ERROR] Combined dataset not found!")
        print("Please check processing output above.")
        return False
    
    # Step 3: Train model
    print("\n" + "="*70)
    print("[STEP 2] Training model on combined dataset...")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, "train_ml_model.py"],
            timeout=3600  # 1 hour max
        )
        
        if result.returncode == 0:
            print("\n[SUCCESS] Model training complete!")
            return True
        else:
            print("\n[ERROR] Training failed")
            return False
    except subprocess.TimeoutExpired:
        print("[WARNING] Training is taking longer than expected.")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

