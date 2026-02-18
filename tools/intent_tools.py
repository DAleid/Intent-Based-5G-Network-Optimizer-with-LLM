"""
Intent Tools — Agent 1: Intent Interpreter
===========================================
Uses the real Groq LLM (Llama 3.3 70B) to parse natural language user input
into a structured intent dictionary that the rest of the agent pipeline uses.

The LLM acts as a TRANSLATOR only — it never controls the network directly.
Safety validation happens downstream in the rule-based Agent 3 (Validator).
"""

import json
import re
import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import SystemMessage, HumanMessage

# ── Valid intent types the system recognises ──────────────────────────────
VALID_INTENT_TYPES = [
    "stadium_event",
    "concert",
    "emergency",
    "iot_deployment",
    "healthcare",
    "transportation",
    "smart_factory",
    "video_conferencing",
    "gaming",
    "general_optimization",
]

# ── System prompt — tells the LLM exactly what JSON to return ─────────────
INTENT_SYSTEM_PROMPT = """You are an expert 5G network intent parser for a telecom NOC (Network Operations Center).

Your ONLY job is to read a user's natural language request and return a structured JSON object.
You must NEVER return anything except valid JSON — no explanations, no markdown, no code fences.

INTENT TYPES you must classify into (choose the single best match):
- stadium_event    : large outdoor events, sports matches, concerts in stadiums
- concert          : music concerts, festivals, live entertainment
- emergency        : emergency services, ambulance, police, fire, disaster response
- iot_deployment   : IoT sensors, smart devices, machine-to-machine, SCADA
- healthcare       : hospitals, remote surgery, medical monitoring
- transportation   : connected vehicles, autonomous driving, V2X, trains
- smart_factory    : industrial automation, robots, factory floor
- video_conferencing: business video calls, webinars, remote meetings
- gaming           : online gaming, esports, AR/VR gaming
- general_optimization: anything that does not fit the above

APPLICATION TYPES (for entities.application):
- video_streaming, voice, data, iot, mixed, gaming, video_conferencing

PRIORITY LEVELS: critical, high, normal, low

SLICE TYPES: eMBB, URLLC, mMTC
- eMBB  → high bandwidth use cases (streaming, gaming, conferencing)
- URLLC → ultra-low latency (emergency, healthcare, robotics, V2X)
- mMTC  → massive device count, low bandwidth (IoT, sensors)

Return EXACTLY this JSON structure (no extra fields, no comments):
{
  "intent_type": "<one of the 10 types above>",
  "slice_type": "<eMBB | URLLC | mMTC>",
  "confidence": <float between 0.0 and 1.0>,
  "entities": {
    "expected_users": <integer — estimate if not given>,
    "application": "<application type>",
    "priority": "<critical | high | normal | low>",
    "bandwidth_mbps": <integer — estimate appropriate bandwidth>,
    "latency_target_ms": <integer — target latency in ms>,
    "time_window": "<immediate | scheduled | unknown>",
    "location_hint": "<brief location description or unknown>"
  },
  "raw_intent": "<the original user text, unchanged>",
  "parsed_successfully": true
}

ESTIMATION GUIDELINES for expected_users if not stated:
- Stadium / Concert: 30000–80000
- Emergency: 50–500
- IoT deployment: 1000–50000 (devices, not people)
- Healthcare: 10–200
- Smart factory: 100–2000 (devices)
- Gaming / Video conferencing: 100–5000
- Transportation: 500–10000
- General: 1000

CONFIDENCE scoring:
- 0.95+ : user gave explicit numbers, clear use case
- 0.85–0.94 : clear use case, some estimation needed
- 0.70–0.84 : context understood but vague details
- below 0.70 : very ambiguous or unclear request"""


def _parse_intent_impl(user_intent: str) -> dict:
    """
    Parse natural language intent using the real Groq LLM.

    Returns a structured dict compatible with the rest of the agent pipeline.
    Falls back to a safe default if the LLM call fails for any reason.
    """
    if not user_intent or not user_intent.strip():
        return _fallback_intent("empty_input", "Empty input provided")

    try:
        from agents.llm_client import get_llm
        llm = get_llm(temperature=0.0)   # 0 = deterministic JSON output

        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"Parse this 5G network intent request:\n\n{user_intent.strip()}"),
        ]

        response = llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip markdown code fences if the model adds them despite instructions
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

        parsed = json.loads(raw_text)

        # ── Validate and sanitise the returned JSON ──────────────────────
        intent_type = parsed.get("intent_type", "general_optimization")
        if intent_type not in VALID_INTENT_TYPES:
            intent_type = "general_optimization"

        confidence = float(parsed.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))   # clamp to [0, 1]

        entities = parsed.get("entities", {})
        expected_users = int(entities.get("expected_users", 1000))
        expected_users = max(1, min(500_000, expected_users))  # apply hard limits

        bandwidth = int(entities.get("bandwidth_mbps", 100))
        bandwidth = max(10, min(500, bandwidth))               # apply hard limits

        latency = int(entities.get("latency_target_ms", 30))
        latency = max(1, min(500, latency))

        return {
            "intent_type": intent_type,
            "slice_type": parsed.get("slice_type", _default_slice(intent_type)),
            "confidence": confidence,
            "entities": {
                "expected_users": expected_users,
                "application": entities.get("application", "mixed"),
                "priority": entities.get("priority", "normal"),
                "bandwidth_mbps": bandwidth,
                "latency_target_ms": latency,
                "time_window": entities.get("time_window", "unknown"),
                "location_hint": entities.get("location_hint", "unknown"),
            },
            "raw_intent": user_intent,
            "parsed_successfully": True,
            "llm_powered": True,         # flag so the UI can show "AI-Powered" badge
        }

    except json.JSONDecodeError as e:
        # LLM returned something that isn't valid JSON
        return _fallback_intent(
            user_intent,
            f"LLM returned non-JSON response: {e}",
            use_keyword_fallback=True,
        )
    except Exception as e:
        error_msg = str(e)
        if "GROQ_API_KEY" in error_msg or "api_key" in error_msg.lower():
            # Surface a clear API key error to the user
            raise ValueError(
                "❌ GROQ_API_KEY not set. Please:\n"
                "1. Get a free key at https://console.groq.com\n"
                "2. Add it to your .env file:  GROQ_API_KEY=gsk_...\n"
                "3. Restart the app"
            ) from e
        # Any other error — use keyword fallback so the app keeps running
        return _fallback_intent(user_intent, f"LLM error: {error_msg}", use_keyword_fallback=True)


def _default_slice(intent_type: str) -> str:
    """Return the most appropriate 3GPP slice type for a given intent."""
    urllc_intents = {"emergency", "healthcare", "transportation"}
    mmtc_intents  = {"iot_deployment"}
    if intent_type in urllc_intents:
        return "URLLC"
    if intent_type in mmtc_intents:
        return "mMTC"
    return "eMBB"


def _fallback_intent(user_intent: str, reason: str, use_keyword_fallback: bool = False) -> dict:
    """
    Return a safe default intent when the LLM is unavailable.

    If use_keyword_fallback=True, tries simple keyword matching first
    so the app degrades gracefully rather than crashing.
    """
    intent_type = "general_optimization"
    confidence  = 0.60

    if use_keyword_fallback and isinstance(user_intent, str):
        text = user_intent.lower()
        keyword_map = {
            "stadium_event":       ["stadium", "match", "football", "soccer", "crowd"],
            "concert":             ["concert", "music", "festival", "band", "show"],
            "emergency":           ["emergency", "urgent", "ambulance", "fire", "police", "disaster"],
            "iot_deployment":      ["iot", "sensor", "device", "smart meter", "scada"],
            "healthcare":          ["hospital", "surgery", "medical", "patient", "clinic"],
            "transportation":      ["vehicle", "car", "train", "autonomous", "v2x", "highway"],
            "smart_factory":       ["factory", "robot", "industrial", "automation", "machine"],
            "video_conferencing":  ["conference", "meeting", "webinar", "teams", "zoom"],
            "gaming":              ["gaming", "esport", "game", "vr", "ar", "play"],
        }
        for intent, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                intent_type = intent
                confidence  = 0.72
                break

    slice_type = _default_slice(intent_type)

    return {
        "intent_type": intent_type,
        "slice_type": slice_type,
        "confidence": confidence,
        "entities": {
            "expected_users": 1000,
            "application": "mixed",
            "priority": "normal",
            "bandwidth_mbps": 100,
            "latency_target_ms": 30,
            "time_window": "unknown",
            "location_hint": "unknown",
        },
        "raw_intent": user_intent,
        "parsed_successfully": True,
        "llm_powered": False,
        "fallback_reason": reason,
    }


# ── CrewAI-compatible Tool wrapper (used if you enable CrewAI agents) ─────

try:
    from crewai.tools import tool as crewai_tool

    @crewai_tool("Parse Network Intent")
    def parse_intent(user_intent: str) -> str:
        """
        Parse a natural language 5G network optimization intent.
        Returns a JSON string with intent_type, confidence, and entities.
        """
        result = _parse_intent_impl(user_intent)
        return json.dumps(result, indent=2)

except ImportError:
    # CrewAI not installed — plain function used directly by app.py
    def parse_intent(user_intent: str) -> dict:
        """Parse a natural language 5G network intent. Returns a dict."""
        return _parse_intent_impl(user_intent)
