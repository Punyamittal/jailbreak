"""
Run test on jailbreak dataset and save output to file.
"""

import json
import pickle
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Redirect output to file as well
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

output_file = open("jailbreak_test_output.txt", 'w', encoding='utf-8')
sys.stdout = TeeOutput(sys.stdout, output_file)

try:
    print("="*70)
    print("TESTING BALANCED MODEL ON JAILBREAK DATASET")
    print("="*70)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
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
    
    print(f"  Loaded {len(prompts)} examples")
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Load model
    print("\n[2/3] Loading balanced model...")
    from train_balanced_model import BalancedAntiJailbreakModel
    model = BalancedAntiJailbreakModel()
    
    if Path("models/balanced_ensemble.pkl").exists():
        with open("models/balanced_ensemble.pkl", 'rb') as f:
            model.ensemble_model = pickle.load(f)
            model.model = model.ensemble_model
        print("  [OK] Ensemble model")
    else:
        with open("models/balanced_model.pkl", 'rb') as f:
            model.model = pickle.load(f)
        print("  [OK] Regular model")
    
    with open("models/balanced_vectorizer.pkl", 'rb') as f:
        model.vectorizer = pickle.load(f)
    with open("models/balanced_encoder.pkl", 'rb') as f:
        model.label_encoder = pickle.load(f)
    model.is_trained = True
    
    # Test
    print(f"\n[3/3] Testing {len(prompts)} prompts...")
    predictions = []
    confidences = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(prompts)}...", flush=True)
        
        try:
            result = model.predict(prompt)
            predictions.append(result['label'])
            confidences.append(result['confidence'])
        except Exception as e:
            print(f"  Error {i+1}: {e}", flush=True)
            predictions.append('benign')
            confidences.append(0.0)
    
    # Metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    recall = recall_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    f1 = f1_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=['benign', 'jailbreak'])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%} (target: >80%)")
    print(f"F1:        {f1:.2%}")
    print(f"\nFalse Negatives: {fn} (rate: {fnr:.2%}, target: <20%)")
    print(f"False Positives:  {fp} (rate: {fpr:.2%})")
    print(f"\nConfusion Matrix:")
    print(f"            Predicted")
    print(f"          Benign  Jailbreak")
    print(f"Benign      {tn:4d}    {fp:4d}")
    print(f"Jailbreak   {fn:4d}    {tp:4d}")
    
    if recall >= 0.80 and fnr < 0.20:
        print("\n✅ MEETS SECURITY REQUIREMENTS")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
    
    # Save JSON
    with open("test_results_jailbreak.json", 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'false_negative_rate': float(fnr),
            'false_positive_rate': float(fpr)
        }, f, indent=2)
    
    print("\n[OK] Results saved to test_results_jailbreak.json")
    print("[OK] Output saved to jailbreak_test_output.txt")
    
finally:
    output_file.close()
    sys.stdout = sys.__stdout__

