# Quick Test Instructions for Jailbreak Dataset

## Step 1: Save the Dataset

You provided a large JSON dataset. Save it to a file:

**Option A: Save directly**
1. Copy the entire JSON array from your query
2. Save it as `jailbreak_test_dataset.json` in the project directory

**Option B: Use Python**
```python
# Create a file called 'my_dataset.json' with your JSON data, then:
python save_dataset_from_query.py my_dataset.json
```

## Step 2: Test the Model

Once the dataset is saved, run:

```bash
python test_on_jailbreak_dataset.py
```

## What to Expect

The test will:
1. Load the dataset (should have ~100+ examples)
2. Load the trained intent model
3. Test each prompt
4. Calculate metrics:
   - Overall accuracy
   - Jailbreak recall (target: >80%)
   - False negative rate (target: <20%)
   - False positive rate
   - Confusion matrix
5. Show sample predictions (false negatives, false positives, high confidence correct)

## Expected Output

```
======================================================================
TESTING INTENT MODEL ON JAILBREAK DATASET
======================================================================

[1/3] Loading dataset from jailbreak_test_dataset.json...
  Loaded 100 examples

  Label distribution:
    jailbreak: 50 (50.0%)
    benign: 50 (50.0%)

[2/3] Loading intent model...
  [OK] Model loaded from models/intent_detector.pkl

[3/3] Testing model...
Testing 100 prompts...
  Processed 50/100...
  Processed 100/100...

======================================================================
TEST RESULTS
======================================================================

📊 Overall Metrics:
  Accuracy:        85.00%
  Precision:       82.00%
  Recall:          88.00%
  F1-Score:        84.90%

🎯 Security-Critical Metrics:
  Jailbreak Recall: 88.00% (must be >80%) ✅
  False Negatives:  6 (jailbreaks missed)
  False Negative Rate: 12.00% (must be <20%) ✅

...
```

## If Model Not Trained

If you see "Model not found", train first:

```bash
python train_intent_model.py
```

This will take 5-10 minutes for the first run (model download + training).

