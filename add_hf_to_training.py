"""
Add HuggingFace dataset to training data.

This script loads the HF dataset and adds it to the combined training dataset.
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

def load_hf_dataset():
    """Load HuggingFace dataset."""
    print("="*70)
    print("LOADING HUGGINGFACE DATASET")
    print("="*70)
    
    try:
        print("\nLoading from HuggingFace...")
        df = pd.read_parquet("hf://datasets/SpawnedShoyo/ai-jailbreak/data/train-00000-of-00001.parquet")
        print(f"[OK] Loaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            from datasets import load_dataset
            dataset = load_dataset("SpawnedShoyo/ai-jailbreak", split="train")
            df = dataset.to_pandas()
            print(f"[OK] Loaded {len(df):,} rows via datasets library")
            return df
        except Exception as e2:
            print(f"[ERROR] {e2}")
            return None


def process_hf_dataset(df):
    """Process HF dataset and convert to training format."""
    print("\n" + "="*70)
    print("PROCESSING HF DATASET")
    print("="*70)
    
    # Find columns
    prompt_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'prompt' in col_lower:
            prompt_col = col
        if 'label' in col_lower:
            label_col = col
    
    if not prompt_col:
        prompt_col = df.columns[0]
    
    print(f"\nUsing columns:")
    print(f"  Prompt: {prompt_col}")
    print(f"  Label: {label_col}")
    
    # Process data
    processed_data = []
    
    for idx, row in df.iterrows():
        prompt = str(row[prompt_col]).strip()
        if not prompt or len(prompt) < 5:
            continue
        
        # Convert label
        if label_col:
            label_val = row[label_col]
            if isinstance(label_val, (int, float)):
                label = 'jailbreak' if label_val > 0.5 else 'benign'
            elif isinstance(label_val, str):
                label_lower = str(label_val).lower()
                if 'jailbreak' in label_lower or label_lower in ['1', 'true', 'yes']:
                    label = 'jailbreak'
                else:
                    label = 'benign'
            else:
                label = 'benign'  # Default
        else:
            label = 'benign'  # Default if no label
        
        processed_data.append({
            "prompt": prompt,
            "label": label,
            "source": "hf_ai_jailbreak",
            "original_label": str(label_val) if label_col else "unknown"
        })
    
    print(f"\nProcessed {len(processed_data):,} examples")
    
    # Label distribution
    labels = Counter(item['label'] for item in processed_data)
    print(f"\nLabel distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count:,} ({count/len(processed_data)*100:.1f}%)")
    
    return processed_data


def add_to_combined_dataset(hf_data, output_path="datasets/combined_with_hf.jsonl"):
    """Add HF data to combined dataset."""
    print("\n" + "="*70)
    print("COMBINING WITH EXISTING DATASET")
    print("="*70)
    
    # Load existing combined dataset
    combined_path = Path("datasets/combined_training_dataset.jsonl")
    all_data = []
    
    if combined_path.exists():
        print(f"\nLoading existing dataset: {combined_path}")
        with open(combined_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
        print(f"  Loaded {len(all_data):,} existing examples")
    else:
        print(f"\n[WARNING] Combined dataset not found, using only HF data")
    
    # Add HF data
    print(f"\nAdding {len(hf_data):,} HF examples...")
    all_data.extend(hf_data)
    
    # Save combined
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistics
    labels = Counter(item['label'] for item in all_data)
    print(f"\nFinal combined dataset:")
    print(f"  Total examples: {len(all_data):,}")
    for label, count in labels.items():
        print(f"    {label}: {count:,} ({count/len(all_data)*100:.1f}%)")
    
    return output_path


def main():
    """Main function."""
    # Load HF dataset
    df = load_hf_dataset()
    if df is None:
        return
    
    # Process HF dataset
    hf_data = process_hf_dataset(df)
    
    # Add to combined dataset
    combined_path = add_to_combined_dataset(hf_data)
    
    print("\n" + "="*70)
    print("[SUCCESS] HF DATA ADDED TO TRAINING DATASET!")
    print("="*70)
    print(f"\nNew dataset: {combined_path}")
    print(f"\nNext step: Retrain model on this dataset:")
    print(f"  python train_balanced_model.py")
    print(f"  (Update dataset_path to: {combined_path})")


if __name__ == "__main__":
    main()

