"""
Add custom dataset to training data.

This script loads the custom_test_data.csv and adds it to the training dataset.
"""

import csv
import json
from pathlib import Path
from collections import Counter

def load_custom_dataset(csv_path: str = "custom_test_data.csv"):
    """Load and process custom dataset."""
    print("="*70)
    print("LOADING CUSTOM DATASET")
    print("="*70)
    
    if not Path(csv_path).exists():
        print(f"[ERROR] {csv_path} not found!")
        return None
    
    print(f"\nLoading from {csv_path}...")
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip('"')
            violates = row['violates'].strip().lower() == 'true'
            
            data.append({
                'prompt': text,
                'label': 'jailbreak' if violates else 'benign',
                'source': 'custom_test_dataset',
                'violates': violates
            })
    
    print(f"[OK] Loaded {len(data)} examples")
    
    # Label distribution
    labels = Counter(item['label'] for item in data)
    print(f"\nLabel distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
    
    return data


def add_to_training_dataset(custom_data, output_path="datasets/combined_with_custom.jsonl"):
    """Add custom data to existing training dataset."""
    print("\n" + "="*70)
    print("COMBINING WITH TRAINING DATASET")
    print("="*70)
    
    # Try to load existing combined dataset (with HF if available)
    existing_paths = [
        "datasets/combined_with_hf.jsonl",
        "datasets/combined_training_dataset.jsonl"
    ]
    
    all_data = []
    used_path = None
    
    for path in existing_paths:
        if Path(path).exists():
            print(f"\nLoading existing dataset: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line))
                    except:
                        continue
            used_path = path
            print(f"  Loaded {len(all_data):,} existing examples")
            break
    
    if not all_data:
        print("\n[WARNING] No existing dataset found. Using only custom data.")
    
    # Add custom data
    print(f"\nAdding {len(custom_data)} custom examples...")
    all_data.extend(custom_data)
    
    # Save combined dataset
    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistics
    labels = Counter(item['label'] for item in all_data)
    sources = Counter(item.get('source', 'unknown') for item in all_data)
    
    print(f"\nFinal combined dataset:")
    print(f"  Total examples: {len(all_data):,}")
    print(f"\n  Label distribution:")
    for label, count in labels.items():
        print(f"    {label}: {count:,} ({count/len(all_data)*100:.1f}%)")
    
    print(f"\n  Source distribution:")
    for source, count in sources.most_common(10):
        print(f"    {source}: {count:,}")
    
    return output_path


def main():
    """Main function."""
    # Load custom dataset
    custom_data = load_custom_dataset()
    if custom_data is None:
        return
    
    # Add to training dataset
    combined_path = add_to_training_dataset(custom_data)
    
    print("\n" + "="*70)
    print("[SUCCESS] CUSTOM DATASET ADDED TO TRAINING!")
    print("="*70)
    print(f"\nNew dataset: {combined_path}")
    print(f"\nNext steps:")
    print(f"  1. Retrain model on this dataset:")
    print(f"     python train_balanced_model.py")
    print(f"     (Update dataset_path to: {combined_path})")
    print(f"\n  2. Or use train_with_threshold.py:")
    print(f"     python train_with_threshold.py")
    print(f"     (It will automatically use the latest combined dataset)")


if __name__ == "__main__":
    main()

