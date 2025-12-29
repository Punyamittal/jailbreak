"""
Quick test on HuggingFace dataset - simplified version.
"""

import pandas as pd
import sys
from pathlib import Path
from collections import Counter

print("="*70)
print("QUICK HF DATASET TEST")
print("="*70)

# Step 1: Load dataset
print("\n[1/3] Loading HuggingFace dataset...")
try:
    df = pd.read_parquet("hf://datasets/SpawnedShoyo/ai-jailbreak/data/train-00000-of-00001.parquet")
    print(f"[OK] Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row sample:")
    print(df.iloc[0].to_dict())
except Exception as e:
    print(f"[ERROR] {e}")
    print("\nTrying alternative...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("SpawnedShoyo/ai-jailbreak", split="train")
        df = dataset.to_pandas()
        print(f"[OK] Loaded {len(df):,} rows via datasets library")
    except Exception as e2:
        print(f"[ERROR] {e2}")
        sys.exit(1)

# Step 2: Load model
print("\n[2/3] Loading trained model...")
try:
    from train_balanced_model import BalancedAntiJailbreakModel
    import pickle
    
    model = BalancedAntiJailbreakModel()
    with open("models/balanced_model.pkl", 'rb') as f:
        model.model = pickle.load(f)
    with open("models/balanced_vectorizer.pkl", 'rb') as f:
        model.vectorizer = pickle.load(f)
    with open("models/balanced_encoder.pkl", 'rb') as f:
        model.label_encoder = pickle.load(f)
    model.is_trained = True
    print("[OK] Model loaded")
except Exception as e:
    print(f"[ERROR] {e}")
    print("Make sure you've trained the model first!")
    sys.exit(1)

# Step 3: Test on sample
print("\n[3/3] Testing on sample data...")

# Find prompt column
prompt_col = None
for col in df.columns:
    if 'prompt' in col.lower() or 'text' in col.lower() or 'input' in col.lower():
        prompt_col = col
        break

if not prompt_col:
    prompt_col = df.columns[0]  # Use first column

print(f"Using column: {prompt_col}")

# Test on first 100 examples (for better accuracy estimate)
test_count = min(100, len(df))
print(f"\nTesting on {test_count} examples:")

predictions = []
confidences = []
actual_labels = []

# Check if we have labels
label_col = None
for col in df.columns:
    if 'label' in col.lower() or 'jailbreak' in col.lower() or 'target' in col.lower():
        label_col = col
        break

for i in range(test_count):
    prompt = str(df.iloc[i][prompt_col])
    result = model.predict(prompt)
    
    predictions.append(result['label'])
    confidences.append(result['confidence'])
    
    if label_col:
        actual_labels.append(df.iloc[i][label_col])
    
    if i < 5:  # Show first 5 in detail
        print(f"\n  Example {i+1}:")
        print(f"    Prompt: {prompt[:60]}...")
        print(f"    Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
        if label_col:
            actual = df.iloc[i][label_col]
            normalized = 'jailbreak' if (isinstance(actual, (int, float)) and actual > 0.5) or (isinstance(actual, str) and 'jailbreak' in str(actual).lower()) else 'benign'
            match = "✓" if result['label'] == normalized else "✗"
            print(f"    Actual: {normalized} {match}")

# Calculate accuracy if labels available
if label_col and len(actual_labels) > 0:
    print(f"\n" + "="*70)
    print("ACCURACY CALCULATION")
    print("="*70)
    
    # Normalize labels
    normalized_actuals = []
    for label in actual_labels:
        if isinstance(label, (int, float)):
            normalized_actuals.append('jailbreak' if label > 0.5 else 'benign')
        elif isinstance(label, str):
            label_lower = str(label).lower()
            if 'jailbreak' in label_lower or label_lower in ['1', 'true', 'yes']:
                normalized_actuals.append('jailbreak')
            else:
                normalized_actuals.append('benign')
        else:
            normalized_actuals.append('unknown')
    
    # Calculate accuracy
    correct = sum(1 for p, a in zip(predictions, normalized_actuals) if p == a and a != 'unknown')
    total = sum(1 for a in normalized_actuals if a != 'unknown')
    
    if total > 0:
        accuracy = correct / total
        print(f"\n📊 ACCURACY: {accuracy:.2%}")
        print(f"   Correct: {correct}/{total}")
        print(f"   Incorrect: {total - correct}/{total}")
        
        # Per-class breakdown
        from collections import Counter
        benign_correct = sum(1 for p, a in zip(predictions, normalized_actuals) if p == 'benign' and a == 'benign')
        benign_total = sum(1 for a in normalized_actuals if a == 'benign')
        jailbreak_correct = sum(1 for p, a in zip(predictions, normalized_actuals) if p == 'jailbreak' and a == 'jailbreak')
        jailbreak_total = sum(1 for a in normalized_actuals if a == 'jailbreak')
        
        print(f"\n   Per-Class:")
        if benign_total > 0:
            print(f"     Benign: {benign_correct}/{benign_total} ({benign_correct/benign_total:.2%})")
        if jailbreak_total > 0:
            print(f"     Jailbreak: {jailbreak_correct}/{jailbreak_total} ({jailbreak_correct/jailbreak_total:.2%})")
    else:
        print("\n[WARNING] Could not calculate accuracy - no valid labels")
else:
    print(f"\n[INFO] No labels available - showing predictions only")
    pred_counts = Counter(predictions)
    print(f"\nPrediction distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")

print(f"\n[SUCCESS] Test complete!")

