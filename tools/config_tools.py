"""
Config Tools — Agent 4: Planner & Configurator
===============================================
Generates 3GPP Release 18 compliant network configurations from a validated intent.

The LLM generates the full configuration (slice type, bandwidth, latency, RAN params)
based on the intent type and entities. All values are clamped to safe ranges by
hard-coded limits so the LLM cannot produce dangerous configurations.

The Validator agent (Agent 3, rule-based) acts as a second safety gate downstream.
"""

import json
import re
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Safety limits — these clamp LLM-generated values ────────────────────
SAFE_LIMITS = {
    "allocated_bandwidth_mbps": (10, 500),
    "latency_target_ms": (1, 500),
    "active_cells": (1, 50),
    "max_bitrate_dl_mbps": (1, 500),
    "max_bitrate_ul_mbps": (1, 200),
    "priority": (1, 10),
}

# Mapping from slice type to 3GPP SST (Service/Slice Type) value
SST_MAP = {"eMBB": 1, "URLLC": 2, "mMTC": 3}

# Valid choices for specific fields
VALID_MIMO = ["1x1", "2x2", "4x4", "8x8"]
VALID_NUMEROLOGY = [0, 1, 2, 3]
VALID_SCHEDULERS = ["round_robin", "proportional_fair", "strict_priority"]

# ── Fallback template — used ONLY when LLM is unavailable ──────────────
FALLBACK_TEMPLATE = {
    "network_slice": {
        "type": "eMBB",
        "name": "General-eMBB",
        "sst": 1,
        "allocated_bandwidth_mbps": 100,
        "latency_target_ms": 30,
        "priority": 4,
    },
    "qos_parameters": {
        "5qi": 9,
        "arp_priority": 4,
        "packet_delay_budget_ms": 30,
        "packet_error_rate": "1e-2",
        "max_bitrate_dl_mbps": 100,
        "max_bitrate_ul_mbps": 30,
    },
    "ran_configuration": {
        "mimo_layers": "4x4",
        "numerology": 1,
        "active_cells": 6,
        "scheduler": "proportional_fair",
        "carrier_aggregation": False,
        "massive_mimo": False,
    },
}

# ── LLM Config Prompt ──────────────────────────────────────────────────
CONFIG_SYSTEM_PROMPT = """You are a 5G network configuration expert following 3GPP Release 18 standards.

Given a user's intent (type, entities, slice type), generate a COMPLETE network configuration.
You must return ONLY valid JSON — no markdown, no explanations, no code fences.

Return EXACTLY this JSON structure:
{
  "network_slice": {
    "type": "<eMBB | URLLC | mMTC>",
    "name": "<short descriptive name like 'Hajj-eMBB' or 'DroneOps-URLLC'>",
    "sst": <1 for eMBB, 2 for URLLC, 3 for mMTC>,
    "allocated_bandwidth_mbps": <integer 10-500>,
    "latency_target_ms": <integer 1-500>,
    "priority": <integer 1-10, where 1 is highest>
  },
  "qos_parameters": {
    "5qi": <integer — 3GPP 5QI value appropriate for this use case>,
    "arp_priority": <integer 1-10>,
    "packet_delay_budget_ms": <integer — same as latency_target_ms>,
    "packet_error_rate": "<scientific notation like 1e-3 or 1e-5>",
    "max_bitrate_dl_mbps": <integer — downlink bitrate>,
    "max_bitrate_ul_mbps": <integer — uplink bitrate>
  },
  "ran_configuration": {
    "mimo_layers": "<1x1 | 2x2 | 4x4 | 8x8>",
    "numerology": <0 | 1 | 2 | 3>,
    "active_cells": <integer 1-50>,
    "scheduler": "<round_robin | proportional_fair | strict_priority>",
    "carrier_aggregation": <true | false>,
    "massive_mimo": <true | false>
  },
  "rationale": "<2-3 sentence technical explanation of why these values were chosen>"
}

GUIDELINES:
- eMBB (SST=1): High bandwidth, moderate latency. Use for streaming, large crowds, downloads.
  Typical: 100-500 Mbps, 10-50 ms, 4x4 or 8x8 MIMO, proportional_fair scheduler.
- URLLC (SST=2): Ultra-low latency, high reliability. Use for emergency, medical, control.
  Typical: 20-100 Mbps, 1-10 ms, 2x2 MIMO, strict_priority scheduler, numerology 2-3.
- mMTC (SST=3): Massive connections, low bandwidth per device. Use for IoT, sensors.
  Typical: 5-50 Mbps total, 50-500 ms, 1x1 MIMO, round_robin scheduler, numerology 0.

5QI reference values:
- 1: Conversational voice  - 2: Live video streaming  - 3: Real-time gaming
- 4: Non-conversational video  - 5: IMS signaling  - 9: Non-GBR default
- 79: mMTC non-critical  - 82: URLLC discrete automation  - 84: V2X
- 85: Remote control  - 86: Real-time remote control

Scale active_cells based on expected_users:
- <1000 users: 3-6 cells
- 1000-10000 users: 6-15 cells
- 10000-100000 users: 15-30 cells
- >100000 users: 30-50 cells

Set priority based on use case criticality:
- 1: Emergency, healthcare, public safety
- 2: Industrial, transportation, critical infrastructure
- 3-4: Entertainment, business, education
- 5+: General, best-effort"""


# ── Rationale prompt (used when config comes from fallback) ─────────────
RATIONALE_PROMPT = """You are a 5G network planning expert. Write a 2-3 sentence technical rationale
for this network configuration. Be specific about why the slice type, bandwidth, and latency
values are appropriate for this use case. Keep it under 60 words."""


def _clamp(value, field_name):
    """Clamp a numeric value to the safe range for the given field."""
    if field_name in SAFE_LIMITS:
        lo, hi = SAFE_LIMITS[field_name]
        return max(lo, min(hi, int(value)))
    return value


def _sanitize_config(config: dict) -> dict:
    """Ensure all config values are within safe ranges and valid choices."""
    c = copy.deepcopy(config)

    ns = c.get("network_slice", {})
    # Clamp slice values
    if ns.get("type") not in ("eMBB", "URLLC", "mMTC"):
        ns["type"] = "eMBB"
    ns["sst"] = SST_MAP.get(ns.get("type", "eMBB"), 1)
    ns["allocated_bandwidth_mbps"] = _clamp(ns.get("allocated_bandwidth_mbps", 100), "allocated_bandwidth_mbps")
    ns["latency_target_ms"] = _clamp(ns.get("latency_target_ms", 30), "latency_target_ms")
    ns["priority"] = _clamp(ns.get("priority", 4), "priority")
    c["network_slice"] = ns

    qos = c.get("qos_parameters", {})
    qos["packet_delay_budget_ms"] = ns["latency_target_ms"]
    qos["max_bitrate_dl_mbps"] = _clamp(qos.get("max_bitrate_dl_mbps", ns["allocated_bandwidth_mbps"]), "max_bitrate_dl_mbps")
    qos["max_bitrate_ul_mbps"] = _clamp(qos.get("max_bitrate_ul_mbps", 30), "max_bitrate_ul_mbps")
    qos["arp_priority"] = _clamp(qos.get("arp_priority", ns["priority"]), "priority")
    if "5qi" not in qos or not isinstance(qos["5qi"], int):
        qos["5qi"] = 9
    if "packet_error_rate" not in qos:
        qos["packet_error_rate"] = "1e-3"
    c["qos_parameters"] = qos

    ran = c.get("ran_configuration", {})
    if ran.get("mimo_layers") not in VALID_MIMO:
        ran["mimo_layers"] = "4x4"
    if ran.get("numerology") not in VALID_NUMEROLOGY:
        ran["numerology"] = 1
    ran["active_cells"] = _clamp(ran.get("active_cells", 6), "active_cells")
    if ran.get("scheduler") not in VALID_SCHEDULERS:
        ran["scheduler"] = "proportional_fair"
    ran["carrier_aggregation"] = bool(ran.get("carrier_aggregation", False))
    ran["massive_mimo"] = bool(ran.get("massive_mimo", False))
    c["ran_configuration"] = ran

    return c


def _generate_config_with_llm(intent_type: str, entities: dict, slice_type: str) -> dict:
    """
    Ask the LLM to generate a full 3GPP config for the given intent.
    Returns sanitized config dict or None if LLM fails.
    """
    try:
        from agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0.1)

        user_message = (
            f"Generate a 3GPP Release 18 network configuration for:\n"
            f"Intent type: {intent_type.replace('_', ' ').title()}\n"
            f"Slice type: {slice_type}\n"
            f"Expected users: {entities.get('expected_users', 1000)}\n"
            f"Application: {entities.get('application', 'mixed')}\n"
            f"Priority level: {entities.get('priority', 'normal')}\n"
            f"Bandwidth requested: {entities.get('bandwidth_mbps', 'auto')} Mbps\n"
            f"Latency target: {entities.get('latency_target_ms', 'auto')} ms\n"
            f"Location: {entities.get('location_hint', 'unknown')}"
        )

        messages = [
            SystemMessage(content=CONFIG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        config = json.loads(raw)

        # Extract rationale before sanitizing (it's a plain string, not a numeric field)
        rationale = config.pop("rationale", "")

        # Sanitize all numeric/enum values to safe ranges
        config = _sanitize_config(config)
        config["rationale"] = rationale

        return config

    except Exception:
        return None


def _generate_rationale(intent_type: str, config: dict) -> str:
    """
    Ask the LLM to explain why this configuration was chosen.
    Used only for fallback configs where the LLM didn't generate the rationale.
    """
    try:
        from agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0.3)

        slice_type = config["network_slice"]["type"]
        bandwidth = config["network_slice"]["allocated_bandwidth_mbps"]
        latency = config["network_slice"]["latency_target_ms"]

        messages = [
            SystemMessage(content=RATIONALE_PROMPT),
            HumanMessage(content=(
                f"Intent: {intent_type.replace('_', ' ').title()}\n"
                f"Slice: {slice_type}, Bandwidth: {bandwidth} Mbps, "
                f"Latency: {latency} ms, Priority: {config['network_slice']['priority']}"
            )),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception:
        slice_type = config["network_slice"]["type"]
        bandwidth = config["network_slice"]["allocated_bandwidth_mbps"]
        latency = config["network_slice"]["latency_target_ms"]
        return (
            f"{slice_type} slice configured with {bandwidth} Mbps bandwidth "
            f"and {latency} ms latency target for {intent_type.replace('_', ' ')} use case."
        )


def _generate_config_impl(intent_result: dict) -> dict:
    """
    Generate a 3GPP Release 18 compliant network configuration.

    Primary path: LLM generates the full config based on intent + entities.
    Fallback path: Generic template with entity overrides if LLM fails.
    All values are clamped to safe ranges regardless of source.
    """
    intent_type = intent_result.get("intent_type", "general_optimization")
    entities = intent_result.get("entities", {})
    slice_type = intent_result.get("slice_type", "eMBB")

    # ── Primary path: LLM generates config ────────────────────────────
    config = _generate_config_with_llm(intent_type, entities, slice_type)

    if config is not None:
        config["intent_type"] = intent_type
        config["expected_users"] = entities.get("expected_users", 1000)
        config["application"] = entities.get("application", "mixed")
        config["3gpp_release"] = "Release 18 (5G-Advanced)"
        config["generated_by"] = "Planner Agent (LLM-generated, safety-clamped)"
        config["llm_powered"] = True
        return config

    # ── Fallback path: Generic template + entity overrides ────────────
    config = copy.deepcopy(FALLBACK_TEMPLATE)

    # Set slice type from intent
    config["network_slice"]["type"] = slice_type
    config["network_slice"]["sst"] = SST_MAP.get(slice_type, 1)
    config["network_slice"]["name"] = f"{intent_type.replace('_', '-').title()}-{slice_type}"

    # Apply entity overrides within safe ranges
    if "bandwidth_mbps" in entities:
        bw = _clamp(entities["bandwidth_mbps"], "allocated_bandwidth_mbps")
        config["network_slice"]["allocated_bandwidth_mbps"] = bw
        config["qos_parameters"]["max_bitrate_dl_mbps"] = bw

    if "latency_target_ms" in entities:
        lat = _clamp(entities["latency_target_ms"], "latency_target_ms")
        config["network_slice"]["latency_target_ms"] = lat
        config["qos_parameters"]["packet_delay_budget_ms"] = lat

    # Generate rationale for fallback config
    config["rationale"] = _generate_rationale(intent_type, config)

    config["intent_type"] = intent_type
    config["expected_users"] = entities.get("expected_users", 1000)
    config["application"] = entities.get("application", "mixed")
    config["3gpp_release"] = "Release 18 (5G-Advanced)"
    config["generated_by"] = "Planner Agent (fallback template + entity overrides)"
    config["llm_powered"] = False
    return config


# ── CrewAI-compatible Tool wrapper ────────────────────────────────────

try:
    from crewai.tools import tool as crewai_tool

    @crewai_tool("Generate Network Configuration")
    def generate_config(intent_result: str) -> str:
        """
        Generate a 3GPP Release 18 network configuration from a parsed intent.
        Input: JSON string of parsed intent. Returns: JSON string of full config.
        """
        if isinstance(intent_result, str):
            intent_result = json.loads(intent_result)
        result = _generate_config_impl(intent_result)
        return json.dumps(result, indent=2)

    @crewai_tool("Get Configuration Templates")
    def get_templates(intent_type: str = "all") -> str:
        """Return the fallback configuration template as JSON."""
        return json.dumps(FALLBACK_TEMPLATE, indent=2)

except ImportError:
    def generate_config(intent_result: dict) -> dict:
        """Generate a 3GPP network configuration from a parsed intent dict."""
        return _generate_config_impl(intent_result)

    def get_templates(intent_type: str = "all"):
        return FALLBACK_TEMPLATE
