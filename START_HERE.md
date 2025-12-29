# 🚀 Getting Started with Intent-Based Anti-Jailbreak System

## Step-by-Step Guide

### Step 1: Install Dependencies

First, make sure you have all required packages:

```bash
pip install sentence-transformers torch scikit-learn numpy
```

**Or install from requirements:**
```bash
pip install -r requirements.txt
```

**Expected output:**
- Downloads sentence-transformers and PyTorch
- May take 2-5 minutes depending on your internet speed
- No errors = ready to proceed

---

### Step 2: Check Your Training Data

Make sure you have a training dataset. The script will automatically find:
1. `datasets/combined_with_custom.jsonl` (if you added custom data)
2. `datasets/combined_with_hf.jsonl` (if you added HF data)
3. `datasets/combined_training_dataset.jsonl` (default)

**Check if dataset exists:**
```bash
python -c "from pathlib import Path; print('Dataset exists:', Path('datasets/combined_training_dataset.jsonl').exists())"
```

**If no dataset exists:**
- Run `python process_large_datasets.py` first
- Or run `python add_custom_to_training.py` to add your custom data

---

### Step 3: Train the Intent Model

Run the training script:

```bash
python train_intent_model.py
```

**What happens:**
1. **First time only:** Downloads sentence-transformer model (~100MB)
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Location: `~/.cache/huggingface/`
   - Time: 2-5 minutes (one-time download)

2. **Loading dataset:**
   ```
   Loading dataset from datasets/combined_training_dataset.jsonl...
     Loaded 180,184 examples
   ```

3. **Encoding texts to embeddings:**
   ```
   [STAGE 2] Training Intent Embedding Model...
     Encoding texts to embeddings...
     Embedding shape: (180184, 384)
   ```
   - Time: 5-15 minutes (depends on dataset size)

4. **Training classifiers:**
   ```
     Training classifiers...
   ```
   - Time: 5-10 minutes

5. **Test results:**
   ```
     Test Results:
               precision    recall  f1-score   support
       benign       0.XX      0.XX      0.XX      XXXX
    jailbreak       0.XX      0.XX      0.XX      XXXX
   ```

6. **Model saved:**
   ```
   [OK] Intent embedding model trained
   Model saved to: models/intent_detector.pkl
   ```

**Total time:** 15-30 minutes (first time), 10-20 minutes (subsequent runs)

---

### Step 4: Test on Custom Dataset

After training completes, test on your custom dataset:

```bash
python test_intent_on_custom.py
```

**Expected output:**
```
======================================================================
TESTING INTENT MODEL ON CUSTOM DATASET
======================================================================

[1/3] Loading intent model...
[OK] Model loaded

[2/3] Loading custom dataset...
[OK] Loaded 60 examples

[3/3] Testing model...
  Processed 10/60...
  Processed 20/60...
  ...

======================================================================
RESULTS
======================================================================

📊 Overall Accuracy: XX.XX% (XX/60)

Per-Class Accuracy:
  Benign: XX.XX% (XX/31)
  Jailbreak: XX.XX% (XX/29)  ← This should be >80%!

Confusion Matrix:
                Predicted
              Benign  Jailbreak
Actual Benign      XX        XX
Jailbreak          XX        XX

Error Rates:
  False Positive Rate: XX.XX%
  False Negative Rate: XX.XX%  ← This should be <20%!

Key Metrics:
  Jailbreak Recall: XX.XX% (target: >80%)
  False Negative Rate: XX.XX% (target: <20%)
```

---

## 🎯 Success Criteria

Your system is working well if:

✅ **Jailbreak Recall > 80%** - Catches most attacks  
✅ **False Negative Rate < 20%** - Doesn't miss many attacks  
✅ **Overall Accuracy > 70%** - Better than old system (45%)

---

## 🐛 Troubleshooting

### Error: "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### Error: "No training dataset found"
```bash
# Create dataset first
python process_large_datasets.py
```

### Error: "Model download failed"
- Check internet connection
- Try again (downloads are cached after first time)

### Training is slow
- Normal! First time takes 15-30 minutes
- Subsequent runs are faster (model cached)

### Low recall (< 80%)
- Lower threshold: Edit `train_intent_model.py`, change `jailbreak_threshold=0.3`
- Add more training data
- Check dataset quality

---

## 📊 Quick Comparison

### Old System (TF-IDF)
- Accuracy: 45%
- Jailbreak Recall: 20.69%
- False Negatives: 79.31%

### New System (Intent-Based)
- Accuracy: 70-85% (expected)
- Jailbreak Recall: 80%+ (target)
- False Negatives: <20% (target)

---

## 🚀 Next Steps After Training

1. **Test on custom dataset** - See improvement
2. **Compare with old system** - Verify it's better
3. **Fine-tune if needed** - Adjust thresholds
4. **Use in production** - Integrate with your pipeline

---

## 💡 Pro Tips

1. **First run takes longest** - Model download + training
2. **Subsequent runs are faster** - Model is cached
3. **Check recall, not accuracy** - Security-first metric
4. **Lower threshold = more sensitive** - Catches more attacks
5. **Monitor false negatives** - These are missed attacks!

---

## 📝 Summary

**To start training:**
```bash
# 1. Install dependencies
pip install sentence-transformers torch

# 2. Train model
python train_intent_model.py

# 3. Test results
python test_intent_on_custom.py
```

**That's it!** The system will automatically:
- Find your training dataset
- Download the sentence-transformer model (first time)
- Train the intent detection model
- Save the trained model
- Show you the results

Good luck! 🎉

