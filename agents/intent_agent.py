"""
Intent Interpreter Agent for 5G-Advanced Network Optimizer

This agent is responsible for understanding user requests and extracting
structured information from natural language input.
"""

from crewai import Agent
from tools.intent_tools import parse_intent
from tools.config_tools import get_templates
from tools.action_tools import db_query


def create_intent_agent(llm) -> Agent:
    """
    Creates the Intent Interpreter Agent.

    This agent specializes in:
    - Natural language understanding
    - Intent classification
    - Entity extraction (location, time, application, etc.)
    - Mapping intents to 5G-Advanced use cases

    Args:
        llm: Language model instance to use

    Returns:
        Configured CrewAI Agent
    """

    return Agent(
        role="5G-Advanced Intent Interpreter",
        goal="""Accurately understand and interpret user requests for 5G-Advanced
        network optimization. Extract all relevant information including location,
        time, application type, priority level, and quality requirements.""",
        backstory="""You are an expert in natural language processing specialized
        in telecommunications. You have deep knowledge of:

        - 5G-Advanced (3GPP Release 18) use cases and scenarios
        - Network optimization terminology
        - Event types and their network requirements
        - QoS requirements for different applications

        Your job is to understand what the user needs and translate their request
        into structured data that other agents can act upon. You are precise,
        thorough, and always consider the 5G-Advanced context of requests.

        When analyzing intents, you consider:
        - Is this a stadium/concert event? → High capacity eMBB
        - Is this an emergency? → Low latency URLLC
        - Is this IoT deployment? → Massive mMTC
        - What are the timing requirements?
        - What quality level is expected?""",
        tools=[parse_intent, get_templates, db_query],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
