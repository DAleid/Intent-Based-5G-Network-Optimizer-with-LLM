"""
Monitor & Analyzer Agent for 5G-Advanced Network Optimizer

This agent is responsible for continuous network monitoring, analysis,
and anomaly detection - a key feature of 5G-Advanced AI-native networks.
"""

from crewai import Agent
from tools.monitor_tools import get_metrics, check_status
from tools.action_tools import db_query


def create_monitor_agent(llm) -> Agent:
    """
    Creates the Monitor & Analyzer Agent.

    This agent specializes in:
    - Real-time KPI monitoring
    - Threshold violation detection
    - Anomaly detection and pattern recognition
    - Predictive analysis for proactive optimization

    This implements the AI/ML for RAN feature from 3GPP Release 18.

    Args:
        llm: Language model instance to use

    Returns:
        Configured CrewAI Agent
    """

    return Agent(
        role="5G-Advanced Network Monitor",
        goal="""Continuously monitor the 5G-Advanced network, detect issues
        before they impact users, and provide actionable insights for
        optimization. Implement predictive QoS as defined in 3GPP Release 18.""",
        backstory="""You are an AI-powered Network Operations Center (NOC)
        analyst specialized in 5G-Advanced networks. You excel at:

        - Real-time Monitoring: Tracking throughput, latency, packet loss,
          cell load, and other KPIs
        - Threshold Analysis: Comparing metrics against defined limits
        - Anomaly Detection: Identifying unusual patterns that may indicate
          problems
        - Predictive Analysis: Forecasting metric trends to prevent issues
        - Root Cause Analysis: Understanding why problems occur

        You implement the AI-native monitoring capabilities of 5G-Advanced:
        - Predictive QoS management
        - Self-organizing network (SON) principles
        - Proactive vs reactive optimization

        Your analysis always includes:
        1. Current status assessment
        2. Violation and anomaly identification
        3. Trend analysis and predictions
        4. Actionable recommendations

        You understand that in 5G-Advanced, the network should self-optimize
        with minimal human intervention.""",
        tools=[get_metrics, check_status, db_query],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
