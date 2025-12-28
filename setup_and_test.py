"""
Quick setup verification script.

Checks if all dependencies are installed and API key is configured.
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    missing = []
    
    try:
        import dotenv
        print("  [OK] python-dotenv")
    except ImportError:
        missing.append("python-dotenv")
        print("  [MISSING] python-dotenv")
    
    try:
        import google.generativeai
        print("  [OK] google-generativeai")
    except ImportError:
        missing.append("google-generativeai")
        print("  [MISSING] google-generativeai")
    
    # Check our modules
    try:
        from pipeline import AntiJailbreakPipeline
        print("  [OK] AntiJailbreakPipeline")
    except ImportError as e:
        print(f"  [ERROR] AntiJailbreakPipeline: {e}")
        return False
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_api_key():
    """Check if API key is configured."""
    print("\nChecking API key configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("  [MISSING] GEMINI_API_KEY not found in .env file")
        print("\nTo fix:")
        print("  1. Create a .env file in the project root")
        print("  2. Add: GEMINI_API_KEY=your_api_key_here")
        print("  3. Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    else:
        # Show first/last few chars for verification
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  [OK] GEMINI_API_KEY found ({masked})")
        return True


def main():
    """Run setup checks."""
    print("=" * 60)
    print("ANTI-JAILBREAK SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    deps_ok = check_dependencies()
    key_ok = check_api_key()
    
    print()
    print("=" * 60)
    if deps_ok and key_ok:
        print("[SUCCESS] All checks passed! Ready to test.")
        print("\nRun: python test_with_gemini.py")
    else:
        print("[FAILED] Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

