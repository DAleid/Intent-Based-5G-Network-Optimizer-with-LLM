"""
test_groq.py  — Run this FIRST to verify your Groq setup
==========================================================
Run from your project root folder:

    python test_groq.py

It will tell you exactly what is wrong if anything fails.
"""

import os
import sys
from pathlib import Path

print("=" * 55)
print("  Groq API Connection Test")
print("=" * 55)

project_root = Path(__file__).resolve().parent
print(f"\n📁 Project root: {project_root}")

# ── Test 1: .env file ────────────────────────────────────────
env_path = project_root / ".env"
print(f"\n[1] Checking .env file at: {env_path}")
if env_path.exists():
    print("    ✅ .env file found")
else:
    print("    ❌ .env file NOT FOUND")
    print(f"       Create it: copy .env.example to .env")
    print(f"       Then add:  GROQ_API_KEY=gsk_your_key_here")
    sys.exit(1)

# ── Test 2: Load .env ────────────────────────────────────────
print("\n[2] Loading .env...")
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path, override=True)
    print("    ✅ python-dotenv loaded successfully")
except ImportError:
    print("    ❌ python-dotenv not installed")
    print("       Run:  pip install python-dotenv")
    sys.exit(1)

# ── Test 3: API key ──────────────────────────────────────────
print("\n[3] Checking GROQ_API_KEY...")
api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    print("    ❌ GROQ_API_KEY is empty")
    print("       Open your .env file and add:  GROQ_API_KEY=gsk_...")
    sys.exit(1)
elif api_key == "your_groq_api_key_here":
    print("    ❌ GROQ_API_KEY is still the placeholder value")
    print("       Get a real key at https://console.groq.com")
    sys.exit(1)
elif not api_key.startswith("gsk_"):
    print(f"    ⚠️  Key found but doesn't start with 'gsk_': {api_key[:8]}...")
    print("       This might still work — continuing...")
else:
    print(f"    ✅ Key found: {api_key[:8]}...{api_key[-4:]}")

# ── Test 4: langchain_groq installed ────────────────────────
print("\n[4] Checking langchain-groq...")
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    print("    ✅ langchain-groq is installed")
except ImportError as e:
    print(f"    ❌ Import failed: {e}")
    print("       Run:  pip install langchain-groq langchain-core")
    sys.exit(1)

# ── Test 5: Real API call ────────────────────────────────────
print("\n[5] Making a real Groq API call...")
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.0,
    )
    response = llm.invoke([
        SystemMessage(content="You are a test assistant. Reply with only valid JSON."),
        HumanMessage(content='Return exactly this JSON: {"status": "ok", "llm": "groq"}'),
    ])
    raw = response.content.strip()
    print(f"    ✅ API responded: {raw[:80]}")
except Exception as e:
    print(f"    ❌ API call failed: {type(e).__name__}: {e}")
    print("\n    Common causes:")
    print("    - Invalid API key (check console.groq.com)")
    print("    - No internet access")
    print("    - Groq service down (check status.groq.com)")
    sys.exit(1)

# ── Test 6: Test the actual intent parser ───────────────────
print("\n[6] Testing intent parser with a real input...")
sys.path.insert(0, str(project_root))
try:
    from tools.intent_tools import _parse_intent_impl
    result = _parse_intent_impl("Emergency ambulance communications in the hospital area, need ultra-low latency immediately")
    print(f"    ✅ Intent parsed successfully!")
    print(f"       Intent type : {result['intent_type']}")
    print(f"       Confidence  : {result['confidence']:.0%}")
    print(f"       Slice type  : {result['slice_type']}")
    print(f"       LLM powered : {result['llm_powered']}")
    print(f"       Priority    : {result['entities']['priority']}")
    print(f"       Users est.  : {result['entities']['expected_users']}")
except Exception as e:
    print(f"    ❌ Intent parser failed: {type(e).__name__}: {e}")
    sys.exit(1)

# ── All passed ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ ALL TESTS PASSED — Groq LLM is working correctly")
print("  You can now run:  streamlit run app.py")
print("=" * 55 + "\n")
