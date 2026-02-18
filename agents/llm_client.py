"""
LLM client for tools — returns a LangChain-compatible ChatGroq instance
for direct .invoke(messages) use inside intent_tools.py and reasoning_llm.py.

Separate from crew.py's get_llm() which returns a crewai.LLM object for agents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    LLM_PROVIDER,
    LLM_MODELS,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    AGENT_CONFIG,
)


def get_llm(temperature: float = None):
    """
    Returns a LangChain ChatGroq (or ChatOpenAI) instance.
    Used by tools that call llm.invoke(messages) directly.

    Args:
        temperature: Override temperature (e.g. 0.0 for deterministic JSON output).
                     Defaults to AGENT_CONFIG["temperature"] if not provided.
    """
    t = temperature if temperature is not None else AGENT_CONFIG["temperature"]

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name=LLM_MODELS["openai"],
            temperature=t,
        )

    # Default: Groq
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODELS["groq"],
        temperature=t,
    )
