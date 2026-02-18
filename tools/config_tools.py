"""
Config Tools — Agent 4: Planner & Configurator
===============================================
Generates 3GPP Release 18 compliant network configurations from a validated intent.

Design principle (safety-first):
  - Configuration VALUES (bandwidth, latency, slice type) are DETERMINISTIC.
    They come from templates — the LLM never chooses these numbers.
  - The LLM only generates a human-readable RATIONALE string shown in the UI.
    It cannot change any configuration parameter.

This keeps the "LLM translates, Rules execute" architecture intact.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 3GPP Release 18 Configuration Templates ──────────────────────────────
# Each template maps an intent_type to a full network configuration.
# Numbers here are deterministic — no LLM involvement.

CONFIG_TEMPLATES = {
    "stadium_event": {
        "network_slice": {
            "type": "eMBB",
            "name": "Stadium-eMBB",
            "sst": 1,
            "allocated_bandwidth_mbps": 200,
            "latency_target_ms": 20,
            "priority": 2,
        },
        "qos_parameters": {
            "5qi": 2,
            "arp_priority": 2,
            "packet_delay_budget_ms": 20,
            "packet_error_rate": "1e-3",
            "max_bitrate_dl_mbps": 200,
            "max_bitrate_ul_mbps": 50,
        },
        "ran_configuration": {
            "mimo_layers": "8x8",
            "numerology": 1,
            "active_cells": 8,
            "scheduler": "proportional_fair",
            "carrier_aggregation": True,
            "massive_mimo": True,
        },
    },
    "concert": {
        "network_slice": {
            "type": "eMBB",
            "name": "Concert-eMBB",
            "sst": 1,
            "allocated_bandwidth_mbps": 150,
            "latency_target_ms": 30,
            "priority": 3,
        },
        "qos_parameters": {
            "5qi": 2,
            "arp_priority": 3,
            "packet_delay_budget_ms": 30,
            "packet_error_rate": "1e-3",
            "max_bitrate_dl_mbps": 150,
            "max_bitrate_ul_mbps": 80,
        },
        "ran_configuration": {
            "mimo_layers": "4x4",
            "numerology": 1,
            "active_cells": 6,
            "scheduler": "proportional_fair",
            "carrier_aggregation": True,
            "massive_mimo": False,
        },
    },
    "emergency": {
        "network_slice": {
            "type": "URLLC",
            "name": "Emergency-URLLC",
            "sst": 2,
            "allocated_bandwidth_mbps": 50,
            "latency_target_ms": 5,
            "priority": 1,   # Highest priority
        },
        "qos_parameters": {
            "5qi": 82,
            "arp_priority": 1,
            "packet_delay_budget_ms": 5,
            "packet_error_rate": "1e-5",
            "max_bitrate_dl_mbps": 50,
            "max_bitrate_ul_mbps": 50,
        },
        "ran_configuration": {
            "mimo_layers": "2x2",
            "numerology": 3,
            "active_cells": 10,
            "scheduler": "strict_priority",
            "carrier_aggregation": False,
            "massive_mimo": False,
        },
    },
    "iot_deployment": {
        "network_slice": {
            "type": "mMTC",
            "name": "IoT-mMTC",
            "sst": 3,
            "allocated_bandwidth_mbps": 20,
            "latency_target_ms": 100,
            "priority": 5,
        },
        "qos_parameters": {
            "5qi": 79,
            "arp_priority": 5,
            "packet_delay_budget_ms": 100,
            "packet_error_rate": "1e-2",
            "max_bitrate_dl_mbps": 1,
            "max_bitrate_ul_mbps": 1,
        },
        "ran_configuration": {
            "mimo_layers": "1x1",
            "numerology": 0,
            "active_cells": 20,
            "scheduler": "round_robin",
            "carrier_aggregation": False,
            "massive_mimo": False,
        },
    },
    "healthcare": {
        "network_slice": {
            "type": "URLLC",
            "name": "Healthcare-URLLC",
            "sst": 2,
            "allocated_bandwidth_mbps": 80,
            "latency_target_ms": 5,
            "priority": 1,
        },
        "qos_parameters": {
            "5qi": 82,
            "arp_priority": 1,
            "packet_delay_budget_ms": 5,
            "packet_error_rate": "1e-5",
            "max_bitrate_dl_mbps": 80,
            "max_bitrate_ul_mbps": 30,
        },
        "ran_configuration": {
            "mimo_layers": "4x4",
            "numerology": 3,
            "active_cells": 5,
            "scheduler": "strict_priority",
            "carrier_aggregation": True,
            "massive_mimo": False,
        },
    },
    "transportation": {
        "network_slice": {
            "type": "URLLC",
            "name": "Transport-URLLC",
            "sst": 2,
            "allocated_bandwidth_mbps": 60,
            "latency_target_ms": 10,
            "priority": 2,
        },
        "qos_parameters": {
            "5qi": 84,
            "arp_priority": 2,
            "packet_delay_budget_ms": 10,
            "packet_error_rate": "1e-4",
            "max_bitrate_dl_mbps": 60,
            "max_bitrate_ul_mbps": 20,
        },
        "ran_configuration": {
            "mimo_layers": "2x2",
            "numerology": 2,
            "active_cells": 15,
            "scheduler": "strict_priority",
            "carrier_aggregation": True,
            "massive_mimo": False,
        },
    },
    "smart_factory": {
        "network_slice": {
            "type": "URLLC",
            "name": "Factory-URLLC",
            "sst": 2,
            "allocated_bandwidth_mbps": 100,
            "latency_target_ms": 5,
            "priority": 2,
        },
        "qos_parameters": {
            "5qi": 82,
            "arp_priority": 2,
            "packet_delay_budget_ms": 5,
            "packet_error_rate": "1e-5",
            "max_bitrate_dl_mbps": 100,
            "max_bitrate_ul_mbps": 50,
        },
        "ran_configuration": {
            "mimo_layers": "4x4",
            "numerology": 3,
            "active_cells": 10,
            "scheduler": "strict_priority",
            "carrier_aggregation": True,
            "massive_mimo": True,
        },
    },
    "video_conferencing": {
        "network_slice": {
            "type": "eMBB",
            "name": "VideoConf-eMBB",
            "sst": 1,
            "allocated_bandwidth_mbps": 120,
            "latency_target_ms": 30,
            "priority": 3,
        },
        "qos_parameters": {
            "5qi": 4,
            "arp_priority": 3,
            "packet_delay_budget_ms": 30,
            "packet_error_rate": "1e-3",
            "max_bitrate_dl_mbps": 120,
            "max_bitrate_ul_mbps": 60,
        },
        "ran_configuration": {
            "mimo_layers": "4x4",
            "numerology": 1,
            "active_cells": 6,
            "scheduler": "proportional_fair",
            "carrier_aggregation": True,
            "massive_mimo": False,
        },
    },
    "gaming": {
        "network_slice": {
            "type": "eMBB",
            "name": "Gaming-eMBB",
            "sst": 1,
            "allocated_bandwidth_mbps": 100,
            "latency_target_ms": 15,
            "priority": 3,
        },
        "qos_parameters": {
            "5qi": 3,
            "arp_priority": 3,
            "packet_delay_budget_ms": 15,
            "packet_error_rate": "1e-3",
            "max_bitrate_dl_mbps": 100,
            "max_bitrate_ul_mbps": 30,
        },
        "ran_configuration": {
            "mimo_layers": "4x4",
            "numerology": 1,
            "active_cells": 8,
            "scheduler": "proportional_fair",
            "carrier_aggregation": True,
            "massive_mimo": False,
        },
    },
    "general_optimization": {
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
    },
}

# ── LLM Rationale Prompt ──────────────────────────────────────────────────
RATIONALE_PROMPT = """You are a 5G network planning expert writing a brief technical rationale.

Given this network configuration that was just generated, write a 2-3 sentence explanation of:
1. Why this slice type ({slice_type}) was chosen
2. Why these bandwidth ({bandwidth} Mbps) and latency ({latency} ms) values are appropriate
3. One key benefit for the end user

Be specific and technical but understandable. Write in plain sentences, no bullet points.
Keep it under 60 words. Do not start with "I" or "The configuration"."""


def _generate_rationale(intent_type: str, config: dict) -> str:
    """
    Ask the LLM to explain why this configuration was chosen.
    Returns a plain-English string for the UI.
    Falls back to a static message if LLM is unavailable.
    """
    try:
        from agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(temperature=0.3)   # Slight creativity for natural-sounding text

        slice_type = config["network_slice"]["type"]
        bandwidth  = config["network_slice"]["allocated_bandwidth_mbps"]
        latency    = config["network_slice"]["latency_target_ms"]

        prompt = RATIONALE_PROMPT.format(
            slice_type=slice_type,
            bandwidth=bandwidth,
            latency=latency,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=(
                f"Intent type: {intent_type.replace('_', ' ').title()}\n"
                f"Slice: {slice_type}, Bandwidth: {bandwidth} Mbps, "
                f"Latency target: {latency} ms, "
                f"Priority: {config['network_slice']['priority']}"
            )),
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception:
        # Rationale is purely cosmetic — always fall back silently
        slice_type = config["network_slice"]["type"]
        bandwidth  = config["network_slice"]["allocated_bandwidth_mbps"]
        latency    = config["network_slice"]["latency_target_ms"]
        return (
            f"{slice_type} slice configured with {bandwidth} Mbps bandwidth "
            f"and {latency} ms latency target for {intent_type.replace('_', ' ')} use case."
        )


def _generate_config_impl(intent_result: dict) -> dict:
    """
    Generate a 3GPP Release 18 compliant network configuration.

    Configuration VALUES are fully deterministic (from templates).
    Only the rationale text is LLM-generated.
    """
    intent_type = intent_result.get("intent_type", "general_optimization")
    entities    = intent_result.get("entities", {})

    # Select base template (fall back to general if unknown intent)
    template = CONFIG_TEMPLATES.get(intent_type, CONFIG_TEMPLATES["general_optimization"])

    # Deep-copy so we never mutate the template
    import copy
    config = copy.deepcopy(template)

    # ── Apply entity overrides (from LLM-parsed intent, bounded by limits) ──
    # These are adjustments WITHIN safe ranges — not free-form LLM output.

    if "bandwidth_mbps" in entities:
        requested_bw = int(entities["bandwidth_mbps"])
        # Honour request but never exceed template max by more than 50%
        template_bw  = config["network_slice"]["allocated_bandwidth_mbps"]
        clamped_bw   = max(10, min(500, min(requested_bw, int(template_bw * 1.5))))
        config["network_slice"]["allocated_bandwidth_mbps"] = clamped_bw
        config["qos_parameters"]["max_bitrate_dl_mbps"]     = clamped_bw

    if "latency_target_ms" in entities:
        requested_lat = int(entities["latency_target_ms"])
        template_lat  = config["network_slice"]["latency_target_ms"]
        # Never allow latency to be HIGHER than template (would degrade QoS)
        clamped_lat   = max(1, min(template_lat, requested_lat))
        config["network_slice"]["latency_target_ms"]            = clamped_lat
        config["qos_parameters"]["packet_delay_budget_ms"]      = clamped_lat

    # ── Generate LLM rationale (cosmetic only) ────────────────────────────
    rationale = _generate_rationale(intent_type, config)
    config["rationale"] = rationale

    # ── Add metadata ──────────────────────────────────────────────────────
    config["intent_type"]      = intent_type
    config["expected_users"]   = entities.get("expected_users", 1000)
    config["application"]      = entities.get("application", "mixed")
    config["3gpp_release"]     = "Release 18 (5G-Advanced)"
    config["generated_by"]     = "Planner Agent (deterministic templates + LLM rationale)"

    return config


# ── CrewAI-compatible Tool wrapper ────────────────────────────────────────

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
        """Return available configuration templates as JSON."""
        if intent_type == "all":
            return json.dumps(list(CONFIG_TEMPLATES.keys()))
        template = CONFIG_TEMPLATES.get(intent_type, CONFIG_TEMPLATES["general_optimization"])
        return json.dumps(template, indent=2)

except ImportError:
    def generate_config(intent_result: dict) -> dict:
        """Generate a 3GPP network configuration from a parsed intent dict."""
        return _generate_config_impl(intent_result)

    def get_templates(intent_type: str = "all"):
        if intent_type == "all":
            return list(CONFIG_TEMPLATES.keys())
        return CONFIG_TEMPLATES.get(intent_type, CONFIG_TEMPLATES["general_optimization"])
