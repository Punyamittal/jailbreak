# Dependency Fixes for Intent-Based Model

## 🔧 Issues Found

1. **NumPy 2.x incompatibility** - sentence-transformers requires NumPy 1.x
2. **Keras 3 incompatibility** - transformers requires tf-keras (Keras 2)

## ✅ Fix Commands

Run these commands to fix all dependency issues:

```bash
# Fix NumPy
pip install "numpy<2" --force-reinstall

# Fix Keras
pip install tf-keras

# Or run the fix script
python fix_dependencies.py
```

## 📋 Complete Installation

For a fresh install, use:

```bash
pip install "numpy<2" tf-keras sentence-transformers torch scikit-learn
```

## ✅ Verify Installation

Test if everything works:

```bash
python -c "import tf_keras; import numpy; print('NumPy:', numpy.__version__); print('tf-keras: OK')"
```

## 🚀 After Fixing

Once dependencies are fixed, you can train:

```bash
python train_intent_model.py
```

## 📝 Updated Requirements

The `requirements.txt` has been updated to include:
- `numpy>=1.21.0,<2.0.0` (NumPy 1.x only)
- `tf-keras>=2.15.0` (for transformers compatibility)

