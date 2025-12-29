# Training Instructions - Security Model

## Current Status

The model training script has been fixed to handle class weights correctly. The training needs to be run to create the model files.

## Quick Start

### Step 1: Verify Setup
```bash
python test_training_setup.py
```

This will check:
- ✅ Relabeled dataset exists
- ✅ LabelEncoder works correctly
- ✅ Class weights are set correctly
- ✅ All imports are available

### Step 2: Train the Model
```bash
python train_security_model.py
```

This will:
1. Load the relabeled dataset (58,489 examples)
2. Balance to 50/50 split (35,186 examples)
3. Train with 8:1 class weights (jailbreak:benign)
4. Use threshold 0.15 for high recall
5. Save to `models/security/`

### Step 3: Evaluate
```bash
python evaluate_security_model.py
```

## Expected Training Time

- **Small dataset** (<50K): 5-10 minutes
- **Medium dataset** (50K-100K): 10-20 minutes
- **Large dataset** (>100K): 20-40 minutes

## What to Expect

### During Training:
```
Loading dataset from datasets/relabeled/all_relabeled_combined.jsonl...
  Loaded 58489 examples
    benign: 17,593 (30.1%)
    jailbreak_attempt: 40,896 (69.9%)

======================================================================
TRAINING SECURITY-FOCUSED JAILBREAK DETECTOR
======================================================================

  Original distribution:
    Benign: 17,593
    Jailbreak: 40,896

  Balanced distribution:
    Benign: 17,593
    Jailbreak: 17,593
    Total: 35,186 examples

  Vectorizing text...
  Feature matrix shape: (35186, 15000)
  Train set: 28,148 examples
  Test set: 7,038 examples

  Class weights: {0: 1, 1: 8}
    benign (idx 0): weight 1
    jailbreak_attempt (idx 1): weight 8

  Training Logistic Regression...
    Recall (CV): 0.85xx (+/- 0.0xxx)

  Training Random Forest...
    Recall (CV): 0.83xx (+/- 0.0xxx)

  Creating ensemble model...
    Recall (CV): 0.87xx (+/- 0.0xxx)

  Evaluating on test set...

  Security Metrics (threshold=0.15):
    Accuracy:  XX.XX%
    Precision: XX.XX%
    Recall:    XX.XX% ⭐ (TARGET: >80%)
    F1-Score:  XX.XX%

  Confusion Matrix:
    [[XXXX XXXX]
     [XXXX XXXX]]

  False Negatives: XXX (rate: XX.XX%, TARGET: <20%)
  False Positives: XXX (rate: XX.XX%)

[OK] Security model saved to models/security/
```

### After Training:
- Model files in `models/security/`:
  - `security_model.pkl` - Trained model
  - `security_vectorizer.pkl` - TF-IDF vectorizer
  - `security_encoder.pkl` - Label encoder
  - `security_threshold.txt` - Threshold value (0.15)

## Troubleshooting

### Error: "Relabeled dataset not found"
**Solution:** Run `python relabel_dataset.py` first

### Error: "ValueError: The classes, [0, 1], are not in class_weight"
**Solution:** This has been fixed. Make sure you're using the latest `train_security_model.py`

### Training takes too long
**Solution:** 
- Reduce `max_features` in vectorizer (currently 15000)
- Reduce `n_estimators` in RandomForest (currently 300)
- Use a smaller sample of the dataset

### Out of memory
**Solution:**
- Reduce dataset size
- Use fewer features
- Close other applications

## Key Features of This Model

1. **Security-Focused**: Optimized for recall (catch most attacks)
2. **Low Threshold**: 0.15 (not 0.5) for high recall
3. **Heavy Class Weights**: 8:1 ratio penalizes false negatives
4. **Hybrid Architecture**: Can combine with rule-based detector
5. **Proper Labels**: Separates policy violations from jailbreak attempts

## Next Steps After Training

1. ✅ Run evaluation: `python evaluate_security_model.py`
2. ✅ Review metrics (recall should be >80%)
3. ✅ Test on your jailbreak dataset
4. ✅ Integrate with your pipeline using `SecurityJailbreakDetector`

## Integration Example

```python
from security_detector import SecurityJailbreakDetector
from train_security_model import SecurityJailbreakModel
import pickle

# Load ML model
ml_model = SecurityJailbreakModel(jailbreak_threshold=0.15)
with open("models/security/security_model.pkl", 'rb') as f:
    ml_model.model = pickle.load(f)
with open("models/security/security_vectorizer.pkl", 'rb') as f:
    ml_model.vectorizer = pickle.load(f)
with open("models/security/security_encoder.pkl", 'rb') as f:
    ml_model.label_encoder = pickle.load(f)
ml_model.is_trained = True

# Create hybrid detector
detector = SecurityJailbreakDetector(
    ml_model=ml_model,
    jailbreak_threshold=0.15
)

# Use
result = detector.predict(user_prompt)
if detector.should_block(result):
    # Block or require confirmation
    print("Jailbreak detected!")
```

