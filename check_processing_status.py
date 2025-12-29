"""
Check the status of dataset processing.
"""

from pathlib import Path
import json

def check_status():
    """Check processing status."""
    print("="*70)
    print("DATASET PROCESSING STATUS")
    print("="*70)
    
    # Check individual processed files
    processed_files = {
        "formatted": "datasets/formatted_processed.jsonl",
        "raw": "datasets/raw_processed.jsonl",
        "synthetic": "datasets/synthetic_processed.jsonl",
        "malignant": "datasets/malignant_labeled.jsonl"
    }
    
    print("\nIndividual datasets:")
    for name, path in processed_files.items():
        if Path(path).exists():
            # Count lines
            with open(path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {count:,} examples ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {name}: Not processed yet")
    
    # Check combined dataset
    combined_path = Path("datasets/combined_training_dataset.jsonl")
    if combined_path.exists():
        print(f"\nCombined dataset:")
        with open(combined_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        size_mb = combined_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Combined: {count:,} examples ({size_mb:.2f} MB)")
        
        # Check label distribution
        from collections import Counter
        labels = Counter()
        with open(combined_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                labels[item['label']] += 1
        
        print(f"\n  Label distribution:")
        for label, count in labels.items():
            print(f"    {label}: {count:,} ({count/sum(labels.values())*100:.1f}%)")
        
        print(f"\n[READY] Combined dataset is ready for training!")
        return True
    else:
        print(f"\n[PENDING] Combined dataset not created yet.")
        print("  Processing may still be running...")
        return False

if __name__ == "__main__":
    check_status()

