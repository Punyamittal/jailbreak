"""
Quick test on custom dataset.
"""

import csv
from pathlib import Path
from collections import Counter

print("="*70)
print("CUSTOM DATASET TEST")
print("="*70)

# Load data
csv_path = Path("custom_test_data.csv")
if not csv_path.exists():
    print(f"[ERROR] {csv_path} not found!")
    exit(1)

print(f"\n[1/3] Loading data from {csv_path}...")
data = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row['text'].strip('"')
        violates = row['violates'].strip().lower() == 'true'
        data.append({
            'text': text,
            'violates': violates,
            'label': 'jailbreak' if violates else 'benign'
        })

print(f"[OK] Loaded {len(data)} examples")

labels = Counter(item['label'] for item in data)
print(f"\nLabel distribution:")
for label, count in labels.items():
    print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")

# Load model
print(f"\n[2/3] Loading model...")
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
    import traceback
    traceback.print_exc()
    exit(1)

# Test
print(f"\n[3/3] Testing model...")
predictions = []
actuals = []
confidences = []

for i, item in enumerate(data):
    try:
        result = model.predict(item['text'])
        predictions.append(result['label'])
        actuals.append(item['label'])
        confidences.append(result['confidence'])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")
    except Exception as e:
        print(f"  Error on {i}: {e}")
        continue

# Results
print(f"\n" + "="*70)
print("RESULTS")
print("="*70)

correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
total = len(predictions)
accuracy = correct / total if total > 0 else 0

print(f"\n📊 Overall Accuracy: {accuracy:.2%} ({correct}/{total})")

# Per-class
benign_correct = sum(1 for p, a in zip(predictions, actuals) if p == 'benign' and a == 'benign')
benign_total = sum(1 for a in actuals if a == 'benign')
benign_acc = benign_correct / benign_total if benign_total > 0 else 0

jailbreak_correct = sum(1 for p, a in zip(predictions, actuals) if p == 'jailbreak' and a == 'jailbreak')
jailbreak_total = sum(1 for a in actuals if a == 'jailbreak')
jailbreak_acc = jailbreak_correct / jailbreak_total if jailbreak_total > 0 else 0

print(f"\nPer-Class:")
print(f"  Benign: {benign_acc:.2%} ({benign_correct}/{benign_total})")
print(f"  Jailbreak: {jailbreak_acc:.2%} ({jailbreak_correct}/{jailbreak_total})")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actuals, predictions, labels=['benign', 'jailbreak'])
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Benign  Jailbreak")
print(f"Actual Benign    {cm[0,0]:5d}      {cm[0,1]:5d}")
print(f"Jailbreak        {cm[1,0]:5d}      {cm[1,1]:5d}")

fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

print(f"\nError Rates:")
print(f"  False Positive Rate: {fp_rate:.2%}")
print(f"  False Negative Rate: {fn_rate:.2%}")

print(f"\n[SUCCESS] Test complete!")

