"""
Reasoning LLM Tools — Agent 2: Reasoner
========================================
Powers two features with real Groq AI:

1. generate_reasoning_questions()
   Generates context-aware clarifying questions based on the detected intent.
   Replaces the static question bank dictionary in app.py.

2. generate_conflict_narration()
   Writes a natural-language negotiation summary for the multi-intent
   conflict resolution tab.

Both functions degrade gracefully to static fallbacks if the LLM is
unavailable, so the app never crashes due to API issues.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── System prompt: clarifying questions ─────────────────────────────────

QUESTIONS_SYSTEM_PROMPT = """You are an expert 5G network consultant helping clarify a user's network request.

Based on the detected intent type and entities, generate exactly 4 clarifying questions
that will help the network planner make better decisions.

Rules:
- Questions must be SPECIFIC to this exact intent type
- Each question must have 3-4 short answer options
- Options should be realistic and mutually exclusive
- Make the questions practical — things that actually change the network config
- Do NOT ask questions already answered by the entities provided

Return ONLY valid JSON — no markdown, no explanations:
[
  {
    "id": "short_snake_case_id",
    "question": "The question text?",
    "options": ["Option A", "Option B", "Option C"],
    "default": "Option A",
    "icon": "single emoji"
  },
  ... (exactly 4 items)
]"""


CONFLICT_SYSTEM_PROMPT = """You are a 5G network resource negotiation expert writing a short summary.

Write a 3-4 sentence paragraph explaining the negotiation outcome in plain language.
Cover: what the conflict was, how priority-based allocation resolved it, and the key trade-off.
Be concise, technical but readable. No bullet points. No headers."""


def generate_reasoning_questions(intent_result: dict) -> list:
    """
    Generate 4 context-aware clarifying questions for the detected intent.

    Returns a list of question dicts compatible with the app.py UI renderer:
    [{"id", "question", "options", "default", "icon"}, ...]

    Falls back to the static question bank if the LLM is unavailable.
    """
    intent_type = intent_result.get("intent_type", "general_optimization")
    entities    = intent_result.get("entities", {})
    confidence  = intent_result.get("confidence", 0.8)

    try:
        from agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0.2)

        user_message = (
            f"Intent type: {intent_type.replace('_', ' ').title()}\n"
            f"Detected entities: {json.dumps(entities, indent=2)}\n"
            f"Confidence: {confidence:.0%}\n\n"
            f"Generate 4 clarifying questions for this specific scenario."
        )

        messages = [
            SystemMessage(content=QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        questions = json.loads(raw)

        # Validate structure — must be a list of 4 dicts with required keys
        validated = []
        required_keys = {"id", "question", "options", "default", "icon"}
        for q in questions:
            if isinstance(q, dict) and required_keys.issubset(q.keys()):
                if isinstance(q["options"], list) and len(q["options"]) >= 2:
                    validated.append(q)

        if len(validated) >= 2:   # Accept if we got at least 2 good questions
            return validated[:4]
        else:
            return _static_questions(intent_type)

    except Exception:
        return _static_questions(intent_type)


def generate_conflict_narration(conflict_data: dict) -> str:
    """
    Generate a natural-language summary of a multi-intent conflict resolution.

    Args:
        conflict_data: dict with keys:
            - intents: list of intent labels
            - total_bandwidth_requested: float (Mbps)
            - available_bandwidth: float (Mbps)
            - resolutions: list of {label, satisfaction, adjustment}

    Returns a plain-text paragraph for display in the UI.
    Falls back to a template string if the LLM is unavailable.
    """
    try:
        from agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0.4)

        intents   = conflict_data.get("intents", [])
        total_bw  = conflict_data.get("total_bandwidth_requested", 0)
        avail_bw  = conflict_data.get("available_bandwidth", 500)
        overload  = ((total_bw / avail_bw) - 1) * 100 if avail_bw > 0 else 0
        resolutions = conflict_data.get("resolutions", [])

        res_summary = "\n".join(
            f"- {r.get('label', '?')}: {r.get('satisfaction', 0):.0f}% satisfied — {r.get('adjustment', '')}"
            for r in resolutions
        )

        user_message = (
            f"Conflict scenario:\n"
            f"Stakeholders: {', '.join(intents)}\n"
            f"Total bandwidth demanded: {total_bw:.0f} Mbps\n"
            f"Available bandwidth: {avail_bw:.0f} Mbps\n"
            f"Network overload: {overload:.0f}%\n\n"
            f"Resolution results:\n{res_summary}\n\n"
            f"Write the negotiation summary."
        )

        messages = [
            SystemMessage(content=CONFLICT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception:
        # Static fallback
        intents  = conflict_data.get("intents", ["multiple stakeholders"])
        total_bw = conflict_data.get("total_bandwidth_requested", 0)
        avail_bw = conflict_data.get("available_bandwidth", 500)
        return (
            f"The system detected a resource conflict where {len(intents)} stakeholders "
            f"collectively requested {total_bw:.0f} Mbps against an available capacity of "
            f"{avail_bw:.0f} Mbps. Priority-based allocation was applied, ensuring "
            f"mission-critical services received full allocation while lower-priority "
            f"services received proportionally adjusted resources."
        )


# ── Static fallback question bank ────────────────────────────────────────
# Used when the LLM is unavailable. Matches the original app.py structure.

def _static_questions(intent_type: str) -> list:
    """Return pre-defined questions for a given intent type."""
    bank = {
        "stadium_event": [
            {"id": "audience_size",   "question": "Expected audience size?",         "options": ["10,000", "30,000", "50,000+"], "default": "30,000", "icon": "👥"},
            {"id": "stream_quality",  "question": "Video stream quality?",           "options": ["HD (1080p)", "4K", "8K"],       "default": "4K",     "icon": "📹"},
            {"id": "urllc_needed",    "question": "Need ultra-low latency comms?",   "options": ["Yes", "No"],                    "default": "No",     "icon": "⚡"},
            {"id": "usage_pattern",   "question": "Usage pattern?",                  "options": ["Peak hours only", "All day"],   "default": "Peak hours only", "icon": "🕐"},
        ],
        "concert": [
            {"id": "audience_size",   "question": "Expected audience size?",         "options": ["5,000", "20,000", "50,000+"],  "default": "20,000", "icon": "👥"},
            {"id": "social_upload",   "question": "Heavy social media uploads?",     "options": ["Yes", "No"],                   "default": "Yes",    "icon": "📱"},
            {"id": "stream_quality",  "question": "Streaming quality required?",     "options": ["HD", "4K", "No streaming"],    "default": "HD",     "icon": "🎥"},
            {"id": "usage_pattern",   "question": "Usage pattern?",                  "options": ["Evening only", "All day"],     "default": "Evening only", "icon": "🕐"},
        ],
        "emergency": [
            {"id": "responder_count", "question": "Number of responders?",           "options": ["<50", "50–200", "200+"],       "default": "50–200", "icon": "🚨"},
            {"id": "urllc_needed",    "question": "Need mission-critical comms?",    "options": ["Yes", "No"],                   "default": "Yes",    "icon": "⚡"},
            {"id": "video_needed",    "question": "Need live video feeds?",          "options": ["Yes", "No"],                   "default": "Yes",    "icon": "📹"},
            {"id": "duration",        "question": "Expected duration?",              "options": ["<1 hour", "1–6 hours", "24h+"], "default": "1–6 hours", "icon": "⏱️"},
        ],
        "healthcare": [
            {"id": "procedure_type",  "question": "Type of medical procedure?",      "options": ["Remote monitoring", "Remote surgery", "Telemedicine"], "default": "Remote monitoring", "icon": "🏥"},
            {"id": "urllc_needed",    "question": "Require ultra-low latency?",      "options": ["Yes (surgery)", "No"],         "default": "Yes (surgery)", "icon": "⚡"},
            {"id": "patient_count",   "question": "Number of patients/sessions?",    "options": ["1–10", "10–50", "50+"],        "default": "1–10",   "icon": "👤"},
            {"id": "guaranteed_qos",  "question": "Need guaranteed QoS?",            "options": ["Yes", "No"],                   "default": "Yes",    "icon": "🛡️"},
        ],
        "iot_deployment": [
            {"id": "device_count",    "question": "Number of IoT devices?",          "options": ["100–1,000", "1,000–10,000", "10,000+"], "default": "1,000–10,000", "icon": "📡"},
            {"id": "data_pattern",    "question": "Data transmission pattern?",      "options": ["Periodic", "Event-driven", "Continuous"], "default": "Periodic", "icon": "📊"},
            {"id": "power_type",      "question": "Device power source?",            "options": ["Battery", "Mains-powered"],     "default": "Battery", "icon": "🔋"},
            {"id": "latency_tolerance","question": "Latency tolerance?",             "options": ["Seconds", "100ms", "<10ms"],    "default": "Seconds", "icon": "⏱️"},
        ],
        "smart_factory": [
            {"id": "device_count",    "question": "Number of connected machines?",   "options": ["50", "200", "1,000+"],         "default": "200",    "icon": "🏭"},
            {"id": "data_pattern",    "question": "Data pattern?",                   "options": ["Periodic telemetry", "Real-time control", "Mixed"], "default": "Mixed", "icon": "📊"},
            {"id": "power_type",      "question": "Device power source?",            "options": ["Battery", "Mains-powered"],     "default": "Mains-powered", "icon": "🔋"},
            {"id": "latency_tolerance","question": "Latency requirement?",           "options": ["Seconds", "100ms", "<10ms"],    "default": "<10ms",  "icon": "⏱️"},
        ],
        "gaming": [
            {"id": "user_count",      "question": "Concurrent gamers?",              "options": ["100", "500", "1,000+"],        "default": "500",    "icon": "🎮"},
            {"id": "quality_priority","question": "Priority?",                       "options": ["Ultra-low latency", "High bandwidth", "Balanced"], "default": "Ultra-low latency", "icon": "🎯"},
            {"id": "guaranteed_qos",  "question": "Need guaranteed QoS?",            "options": ["Yes", "No"],                   "default": "Yes",    "icon": "🛡️"},
            {"id": "usage_pattern",   "question": "Usage pattern?",                  "options": ["Peak hours only", "24/7"],     "default": "Peak hours only", "icon": "🕐"},
        ],
        "video_conferencing": [
            {"id": "user_count",      "question": "Concurrent users?",               "options": ["50", "200", "1,000+"],        "default": "200",    "icon": "📹"},
            {"id": "quality_priority","question": "Quality priority?",               "options": ["Low latency", "4K video", "Balanced"], "default": "4K video", "icon": "🎯"},
            {"id": "guaranteed_qos",  "question": "Need guaranteed QoS?",            "options": ["Yes", "No"],                  "default": "Yes",    "icon": "🛡️"},
            {"id": "usage_pattern",   "question": "Usage pattern?",                  "options": ["Business hours", "All day"],   "default": "Business hours", "icon": "🕐"},
        ],
        "transportation": [
            {"id": "transport_type",  "question": "Transportation type?",            "options": ["Connected vehicles", "Public transit", "Autonomous"], "default": "Connected vehicles", "icon": "🚗"},
            {"id": "coverage_area",   "question": "Coverage scope?",                 "options": ["Highway", "Urban area", "City-wide"], "default": "Urban area", "icon": "📍"},
            {"id": "latency_tolerance","question": "Latency requirement?",           "options": ["50ms", "10ms", "<5ms"],        "default": "10ms",   "icon": "⏱️"},
            {"id": "handover_priority","question": "Seamless handover priority?",    "options": ["High", "Medium", "Low"],       "default": "High",   "icon": "🔄"},
        ],
    }

    default = [
        {"id": "kpi_priority",   "question": "Which KPI to prioritise?",         "options": ["Latency", "Throughput", "Coverage"], "default": "Throughput", "icon": "🎯"},
        {"id": "scope",          "question": "Optimisation scope?",               "options": ["Specific area", "Network-wide"],    "default": "Network-wide", "icon": "📍"},
        {"id": "urgency",        "question": "Urgency level?",                    "options": ["Immediate", "Scheduled", "Best effort"], "default": "Immediate", "icon": "⏱️"},
        {"id": "service_reduction","question": "Accept temporary service reduction?", "options": ["Yes", "No"],                   "default": "No",     "icon": "⚠️"},
    ]

    return bank.get(intent_type, default)
