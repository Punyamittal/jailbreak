# Quick Start: Security Fix

## 🎯 What Was Fixed

Your model was failing because it mixed **policy violations** (drugs, explosives) with **jailbreak attempts** (instruction override). This fix separates them and optimizes for security (high recall).

## 🚀 Run the Fix

### Option 1: Run Everything at Once

```bash
python run_security_fix.py
```

This will:
1. Relabel all datasets
2. Train security-focused model
3. Evaluate on test dataset

### Option 2: Run Step by Step

```bash
# Step 1: Relabel datasets
python relabel_dataset.py

# Step 2: Train security model
python train_security_model.py

# Step 3: Evaluate
python evaluate_security_model.py
```

## 📊 What to Expect

### After Relabeling:
- Dataset split into: `benign`, `policy_violation`, `jailbreak_attempt`
- Saved to: `datasets/relabeled/all_relabeled_combined.jsonl`

### After Training:
- Model optimized for **recall** (target: >80%)
- Low threshold (0.15) for high recall
- Heavy class weights (8:1) to penalize false negatives
- Saved to: `models/security/`

### After Evaluation:
- Performance metrics at different thresholds
- Rule-based vs ML vs Hybrid comparison
- Results saved to: `security_evaluation_results.json`

## ✅ Success Criteria

The model should achieve:
- **Jailbreak Recall**: >80% ⭐
- **False Negative Rate**: <20% ⭐
- **Precision**: ~70-85% (acceptable for security)

## 🔧 Key Files

1. **`security_detector.py`**: Rule-based + ML hybrid detector
2. **`relabel_dataset.py`**: Fixes dataset labels
3. **`train_security_model.py`**: Security-focused training
4. **`evaluate_security_model.py`**: Security metrics
5. **`run_security_fix.py`**: Master script (runs everything)

## 📝 Integration

After training, use in your code:

```python
from security_detector import SecurityJailbreakDetector
from train_security_model import SecurityJailbreakModel
import pickle

# Load model
ml_model = SecurityJailbreakModel(jailbreak_threshold=0.15)
# ... load trained components ...

# Create hybrid detector
detector = SecurityJailbreakDetector(
    ml_model=ml_model,
    jailbreak_threshold=0.15
)

# Use
result = detector.predict(user_prompt)
if detector.should_block(result):
    # Block or require confirmation
    pass
```

## ⚠️ Important

- **Policy violations** are now separate from **jailbreak attempts**
- Model only detects **jailbreak attempts** (instruction override)
- For policy violations, you may want a separate classifier
- Low threshold (0.15) means more false positives (acceptable for security)

## 🎉 Next Steps

1. Run the fix: `python run_security_fix.py`
2. Review results: Check `security_evaluation_results.json`
3. Test manually: `python evaluate_security_model.py`
4. Integrate: Use `SecurityJailbreakDetector` in your pipeline

