"""
Train model with ALL available datasets.
First combines all datasets, then trains the balanced model.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("TRAINING WITH ALL AVAILABLE DATASETS")
    print("="*70)
    
    # Step 1: Combine all datasets
    print("\n[STEP 1/2] Combining all datasets...")
    try:
        result = subprocess.run(
            [sys.executable, "combine_all_datasets.py"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to combine datasets: {result.stderr}")
            return
    except Exception as e:
        print(f"[ERROR] Error combining datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Train model
    print("\n[STEP 2/2] Training balanced model on combined dataset...")
    
    # Update train_balanced_model to use the new dataset
    combined_dataset = Path("datasets/all_datasets_combined.jsonl")
    if not combined_dataset.exists():
        print(f"[ERROR] Combined dataset not found: {combined_dataset}")
        return
    
    # Import and train
    try:
        from train_balanced_model import train_balanced_model
        
        print(f"\nUsing dataset: {combined_dataset}")
        model = train_balanced_model(
            dataset_path=str(combined_dataset),
            balance=True,  # Balance the dataset
            clean=True,    # Clean mislabeled data
            use_ensemble=True  # Use ensemble
        )
        
        print("\n" + "="*70)
        print("[SUCCESS] TRAINING COMPLETE!")
        print("="*70)
        print(f"\nModel saved to: models/")
        print(f"  - balanced_model.pkl")
        print(f"  - balanced_ensemble.pkl")
        print(f"  - balanced_vectorizer.pkl")
        print(f"  - balanced_encoder.pkl")
        print("\nYou can now test the model with:")
        print(f"  python test_jailbreak_dataset.py")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

