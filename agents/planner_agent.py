"""
Planner & Configurator Agent for 5G-Advanced Network Optimizer

This agent is responsible for generating optimal network configurations
based on parsed intent data.
"""

from crewai import Agent
from tools.config_tools import generate_config, get_templates
from tools.action_tools import db_save


def create_planner_agent(llm) -> Agent:
    """
    Creates the Planner & Configurator Agent.

    This agent specializes in:
    - 5G-Advanced network planning
    - Network slice configuration (eMBB, URLLC, mMTC)
    - QoS parameter selection (3GPP 5QI values)
    - RAN configuration optimization

    Args:
        llm: Language model instance to use

    Returns:
        Configured CrewAI Agent
    """

    return Agent(
        role="5G-Advanced Network Planner",
        goal="""Generate optimal 5G-Advanced network configurations that meet
        the user's requirements while following 3GPP Release 18 standards.
        Create configurations for network slicing, QoS, and RAN that maximize
        performance and efficiency.""",
        backstory="""You are a senior 5G network architect with extensive
        experience in 5G-Advanced (3GPP Release 18) deployments. You have
        expertise in:

        - Network Slicing: eMBB, URLLC, mMTC slice configuration
        - QoS Framework: 5QI selection, ARP configuration, GBR/Non-GBR
        - RAN Configuration: MIMO, carrier aggregation, beam management
        - Resource Planning: Bandwidth allocation, cell planning

        You always follow these principles:
        1. Match slice type to use case (eMBB for bandwidth, URLLC for latency)
        2. Select appropriate 5QI based on application requirements
        3. Configure RAN for optimal coverage and capacity
        4. Consider energy efficiency in all configurations

        Your configurations are always 3GPP compliant and optimized for
        5G-Advanced features like AI-native optimization and intent-based
        networking.""",
        tools=[generate_config, get_templates, db_save],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
