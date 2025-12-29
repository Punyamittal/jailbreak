"""
Quick test on jailbreak dataset using balanced model.
"""

import json
import pickle
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
print("Loading dataset...")
with open("jailbreak_test_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = []
labels = []
for item in data:
    prompt = item.get('prompt', '').strip()
    if prompt:
        is_jailbreak = item.get('jailbroken', False)
        labels.append('jailbreak' if is_jailbreak else 'benign')
        prompts.append(prompt)

print(f"Loaded {len(prompts)} examples")
print(f"  Jailbreak: {sum(1 for l in labels if l == 'jailbreak')}")
print(f"  Benign: {sum(1 for l in labels if l == 'benign')}")

# Load model
print("\nLoading balanced model...")
from train_balanced_model import BalancedAntiJailbreakModel
model = BalancedAntiJailbreakModel()

# Load components
if Path("models/balanced_ensemble.pkl").exists():
    with open("models/balanced_ensemble.pkl", 'rb') as f:
        model.ensemble_model = pickle.load(f)
        model.model = model.ensemble_model
    print("  [OK] Ensemble model loaded")
else:
    with open("models/balanced_model.pkl", 'rb') as f:
        model.model = pickle.load(f)
    print("  [OK] Regular model loaded")

with open("models/balanced_vectorizer.pkl", 'rb') as f:
    model.vectorizer = pickle.load(f)
with open("models/balanced_encoder.pkl", 'rb') as f:
    model.label_encoder = pickle.load(f)
model.is_trained = True

# Test
print(f"\nTesting {len(prompts)} prompts...")
predictions = []
confidences = []

for i, prompt in enumerate(prompts):
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(prompts)}...")
    
    try:
        result = model.predict(prompt)
        predictions.append(result['label'])
        confidences.append(result['confidence'])
    except Exception as e:
        print(f"  Error on {i+1}: {e}")
        predictions.append('benign')
        confidences.append(0.0)

# Metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, pos_label='jailbreak', zero_division=0)
recall = recall_score(labels, predictions, pos_label='jailbreak', zero_division=0)
f1 = f1_score(labels, predictions, pos_label='jailbreak', zero_division=0)
cm = confusion_matrix(labels, predictions, labels=['benign', 'jailbreak'])
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nAccuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%} (Jailbreak Recall - target: >80%)")
print(f"F1-Score:  {f1:.2%}")

print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"            Benign  Jailbreak")
print(f"Actual Benign   {tn:4d}    {fp:4d}")
print(f"Actual Jailbreak {fn:4d}    {tp:4d}")

print(f"\nFalse Negatives: {fn} (missed jailbreaks)")
print(f"False Positives: {fp} (benign flagged)")
print(f"False Negative Rate: {fn/(fn+tp)*100:.2f}% (target: <20%)")

if recall >= 0.80 and fn/(fn+tp) < 0.20:
    print("\n✅ MODEL MEETS SECURITY REQUIREMENTS")
else:
    print("\n⚠️  MODEL NEEDS IMPROVEMENT")

