"""
Fix Git Large Files Issue.

This script helps remove large files from Git history and add them to .gitignore.
"""

import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a git command and show output."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd())
        if result.stdout:
            print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Fix large files issue."""
    print("=" * 70)
    print("FIXING GIT LARGE FILES ISSUE")
    print("=" * 70)
    
    # Large files that need to be removed
    large_files = [
        "formatted_dataset.csv",
        "raw_dataset.csv",
        "synthetic_dataset.csv",
        "synthetic_prompts.txt"
    ]
    
    print("\nStep 1: Removing large files from Git index...")
    for file in large_files:
        if Path(file).exists():
            run_command(f'git rm --cached "{file}"', f"Removing {file} from Git")
        else:
            print(f"  {file} not found, skipping...")
    
    print("\nStep 2: Checking .gitignore...")
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if "formatted_dataset.csv" not in content:
            print("  Adding large files to .gitignore...")
            with open(".gitignore", "a") as f:
                f.write("\n# Large dataset files (too big for GitHub)\n")
                for file in large_files:
                    f.write(f"{file}\n")
        else:
            print("  .gitignore already updated")
    else:
        print("  Creating .gitignore...")
        with open(".gitignore", "w") as f:
            f.write("# Large dataset files (too big for GitHub)\n")
            for file in large_files:
                f.write(f"{file}\n")
    
    print("\nStep 3: Staging .gitignore...")
    run_command("git add .gitignore", "Adding .gitignore")
    
    print("\nStep 4: Checking git status...")
    run_command("git status", "Git status")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review the changes: git status")
    print("2. Commit the .gitignore update:")
    print("   git commit -m 'Add large dataset files to .gitignore'")
    print("3. If you need to remove from history (if already pushed):")
    print("   git filter-branch --force --index-filter")
    print("   'git rm --cached --ignore-unmatch formatted_dataset.csv' --prune-empty --tag-name-filter cat -- --all")
    print("4. Force push (if needed): git push origin main --force")
    print("\nWARNING: Force push rewrites history. Only do this if you're sure!")

if __name__ == "__main__":
    main()

