# Quick Start Guide - Testing with Gemini

## Setup (One-Time)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your API Key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Verify Setup

```bash
python setup_and_test.py
```

Should show: `[SUCCESS] All checks passed!`

## Run Tests

```bash
python test_with_gemini.py
```

This will run 6 test scenarios:
1. ✅ Benign prompt (should be allowed)
2. 🚫 Jailbreak attempt (should be blocked)
3. 🚫 Authority escalation (should be blocked)
4. 🚫 Indirect injection (should be blocked)
5. 🚫 Multi-turn escalation (should be detected)
6. ✅ Safe creative request (should be allowed)

## What Happens

1. **Security Pipeline** analyzes every prompt first
2. **Risk Scoring** detects attack patterns
3. **Decision Making** blocks dangerous prompts
4. **Gemini API** only receives safe prompts
5. **Statistics** show security events

## Example Output

```
TEST 2: Jailbreak Attempt (Role-Play)
Decision: BLOCKED
Risk Score: 60.53%
Attack Classes: role_play, instruction_override, authority_escalation
Blocked: Authority escalation attempt detected
```

## Custom Usage

```python
from test_with_gemini import SecureGeminiClient
import os
from dotenv import load_dotenv

load_dotenv()

client = SecureGeminiClient(api_key=os.getenv("GEMINI_API_KEY"))

result = client.generate_content(
    prompt="Your prompt here",
    user_id="user123",
    session_id="session456"
)

if result['success']:
    print(result['response'])
else:
    print(f"Blocked: {result['reason']}")
```

## Files

- `test_with_gemini.py` - Main test script
- `setup_and_test.py` - Setup verification
- `.env` - Your API key (create this)
- `.env.example` - Template for .env file
- `GEMINI_INTEGRATION.md` - Detailed documentation

## Troubleshooting

**Missing dependencies?**
```bash
pip install python-dotenv google-generativeai
```

**API key not found?**
- Create `.env` file
- Add: `GEMINI_API_KEY=your_key`

**Import errors?**
- Make sure you're in the project directory
- Check all Python files are present

## Next Steps

- Integrate into your application
- Customize risk thresholds
- Add monitoring and logging
- Tune attack pattern detection

