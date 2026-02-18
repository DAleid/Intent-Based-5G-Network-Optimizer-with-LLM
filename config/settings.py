"""
Configuration settings for 5G-Advanced Network Optimizer
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LLM Provider Configuration
# =============================================================================
# Options: "groq", "openai", "gemini", "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# API Keys (set in .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model configurations per provider
LLM_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3"
}

# =============================================================================
# 5G-Advanced Network Configuration (3GPP Release 18)
# =============================================================================
NETWORK_CONFIG = {
    "total_bandwidth_mhz": 400,  # Total available bandwidth
    "max_cells": 50,             # Maximum number of cells
    "max_slices": 10,            # Maximum concurrent slices
    "base_latency_ms": 10,       # Base network latency
}

# =============================================================================
# Simulation Settings
# =============================================================================
SIMULATION_CONFIG = {
    "update_interval_seconds": 2,    # How often to update metrics
    "anomaly_probability": 0.15,     # Chance of anomaly occurring
    "max_users_per_cell": 200,       # Maximum users per cell
}

# =============================================================================
# Agent Configuration
# =============================================================================
AGENT_CONFIG = {
    "verbose": True,           # Show agent reasoning
    "max_iterations": 10,      # Max iterations per agent
    "temperature": 0.7,        # LLM temperature
}

# =============================================================================
# Thresholds for Monitoring (5G-Advanced KPIs)
# =============================================================================
KPI_THRESHOLDS = {
    "latency_ms": {
        "target": 50,
        "warning": 80,
        "critical": 100
    },
    "throughput_mbps": {
        "target": 100,
        "warning": 60,
        "critical": 30
    },
    "packet_loss_percent": {
        "target": 0.01,
        "warning": 0.1,
        "critical": 1.0
    },
    "cell_load_percent": {
        "target": 70,
        "warning": 85,
        "critical": 95
    }
}
