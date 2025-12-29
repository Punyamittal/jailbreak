# Large Dataset Files

## Files Not in Git

The following large dataset files are **NOT** tracked in Git because they exceed GitHub's 100 MB file size limit:

- `formatted_dataset.csv` (315.69 MB)
- `raw_dataset.csv` (large)
- `synthetic_dataset.csv` (large)
- `synthetic_prompts.txt` (large)

These files are added to `.gitignore` to prevent accidental commits.

## Using These Datasets

If you need to use these large datasets:

1. **Keep them locally** - They're already in your working directory
2. **Process them** - Use the processing scripts to create smaller, training-ready datasets
3. **Share via other means** - Use cloud storage (Google Drive, Dropbox, etc.) or Git LFS if needed

## Processing Large Datasets

You can process these large datasets using:

```python
# Process and create smaller training datasets
python process_malignant_dataset.py  # For malignant.csv
# Or create custom processing scripts for other datasets
```

## Git LFS (Optional)

If you really need to track these files in Git, you can use Git Large File Storage:

```bash
# Install Git LFS
git lfs install

# Track large CSV files
git lfs track "*.csv"
git lfs track "*.txt"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**Note**: Git LFS requires a GitHub account with LFS quota, and may have costs for large files.

## Recommended Approach

1. **Keep large files local** - Don't commit them
2. **Process into smaller datasets** - Create training-ready JSONL files
3. **Commit processed datasets** - Only commit the smaller, processed files
4. **Document sources** - Note where large datasets came from

The processed datasets in `datasets/` directory are small enough to commit and are already being tracked.

