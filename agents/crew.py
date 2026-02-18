"""
CrewAI Crew Configuration for 5G-Advanced Network Optimizer

This module sets up the multi-agent crew that works together to optimize
5G-Advanced networks based on user intent.
"""

import os
import json
import re
import time
from crewai import Crew, Task, Process, LLM

from .intent_agent import create_intent_agent
from .planner_agent import create_planner_agent
from .monitor_agent import create_monitor_agent
from .optimizer_agent import create_optimizer_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    LLM_PROVIDER,
    LLM_MODELS,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    AGENT_CONFIG
)


def get_llm():
    """
    Get the configured LLM using CrewAI's LLM class (backed by LiteLLM).
    Model must be passed in 'provider/model' format so LiteLLM routes correctly.
    """
    if LLM_PROVIDER == "openai":
        return LLM(
            model=f"openai/{LLM_MODELS['openai']}",
            api_key=OPENAI_API_KEY,
            temperature=AGENT_CONFIG["temperature"]
        )
    # Default: Groq
    return LLM(
        model=f"groq/{LLM_MODELS['groq']}",
        api_key=GROQ_API_KEY,
        temperature=AGENT_CONFIG["temperature"]
    )


def _kickoff_with_retry(crew, max_retries: int = 3):
    """
    Run crew.kickoff() with automatic retry on Groq rate-limit errors.
    Waits 5 s on first retry, 10 s on second, 20 s on third.
    """
    delays = [5, 10, 20]
    for attempt in range(max_retries):
        try:
            return crew.kickoff()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = (
                "rate_limit" in err
                or "ratelimit" in err
                or "rate limit" in err
                or "429" in err
            )
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(delays[attempt])
                continue
            raise


class Network5GOptimizationCrew:
    """
    5G-Advanced Network Optimization Crew

    This crew consists of 4 specialized agents working together:
    1. Intent Interpreter - Understands user requests
    2. Planner & Configurator - Creates network configurations
    3. Monitor & Analyzer - Monitors network performance
    4. Optimizer & Executor - Executes optimizations
    """

    def __init__(self):
        """Initialize the crew with all agents"""
        self.llm = get_llm()

        # Create agents
        self.intent_agent = create_intent_agent(self.llm)
        self.planner_agent = create_planner_agent(self.llm)
        self.monitor_agent = create_monitor_agent(self.llm)
        self.optimizer_agent = create_optimizer_agent(self.llm)

        # Store for tracking
        self.last_intent = None
        self.last_config = None
        self.last_status = None
        self.last_optimization = None

    def create_intent_task(self, user_input: str) -> Task:
        """Create task for intent interpretation"""
        return Task(
            description=f"""Analyze the following user request and extract structured
            intent information:

            User Request: "{user_input}"

            Use the parse_intent tool to analyze this request.
            Extract: intent type, location, time, application, priority, and quality requirements.

            Return a comprehensive analysis that the Planner agent can use to
            generate network configuration.""",
            expected_output="""A structured intent analysis including:
            - Intent type (stadium_event, emergency, iot_deployment, etc.)
            - Extracted entities (location, time, application, priority)
            - Recommended event profile
            - Confidence score""",
            agent=self.intent_agent
        )

    def create_planning_task(self) -> Task:
        """Create task for network planning and configuration"""
        return Task(
            description="""Based on the intent analysis from the previous task,
            generate an optimal 5G-Advanced network configuration.

            Use the generate_config tool with the parsed intent data.

            Create a configuration that includes:
            1. Appropriate network slice (eMBB, URLLC, or mMTC)
            2. QoS parameters following 3GPP Release 18
            3. RAN configuration for optimal performance
            4. Resource allocation plan

            Ensure the configuration is 3GPP compliant and optimized for
            the specific use case.""",
            expected_output="""A complete 5G-Advanced network configuration including:
            - Network slice configuration (type, SST, SD, bandwidth, latency target)
            - QoS parameters (5QI, priority, delay budget, error rate)
            - RAN configuration (cells, MIMO, scheduler)
            - Expected capacity and performance""",
            agent=self.planner_agent
        )

    def create_monitoring_task(self) -> Task:
        """Create task for network monitoring and analysis"""
        return Task(
            description="""Monitor the current network status and analyze performance.

            Steps:
            1. Use get_metrics tool to collect current network KPIs
            2. Use check_status tool to analyze the metrics

            Provide a comprehensive analysis including:
            - Current network health status
            - Any threshold violations
            - Detected anomalies
            - Trend predictions
            - Recommendations for optimization (if needed)""",
            expected_output="""A comprehensive network status report including:
            - Overall health status (healthy/warning/critical)
            - Current KPI values
            - List of violations and anomalies
            - Trend analysis and predictions
            - Specific recommendations if action is needed""",
            agent=self.monitor_agent
        )

    def create_optimization_task(self, recommendation: str = None) -> Task:
        """Create task for executing optimization"""
        description = """Based on the monitoring analysis, execute necessary
        optimizations to improve network performance.

        If the network status shows issues:
        1. Review the recommendations from the monitoring analysis
        2. Select the most appropriate optimization action
        3. Use execute_action tool to perform the optimization
        4. Verify the improvement

        Available actions:
        - scale_bandwidth: For latency or throughput issues
        - activate_cell: For cell load issues
        - adjust_priority: For critical traffic
        - modify_qos: For QoS violations
        - energy_saving: For efficiency optimization"""

        if recommendation:
            description += f"\n\nRecommended action: {recommendation}"

        return Task(
            description=description,
            expected_output="""Optimization execution report including:
            - Action taken (or explanation if no action needed)
            - Before and after metrics
            - Improvement percentages
            - Success verification""",
            agent=self.optimizer_agent
        )

    def run_intent_to_config(self, user_input: str) -> dict:
        """
        Run the intent-to-configuration workflow.

        This is the main workflow for translating user intent into
        network configuration.
        """
        tasks = [
            self.create_intent_task(user_input),
            self.create_planning_task()
        ]

        crew = Crew(
            agents=[self.intent_agent, self.planner_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=AGENT_CONFIG["verbose"]
        )

        result = _kickoff_with_retry(crew)
        return result

    def run_monitoring_cycle(self) -> dict:
        """
        Run a monitoring and optimization cycle.

        This implements the real-time optimization loop of 5G-Advanced.
        """
        tasks = [
            self.create_monitoring_task(),
            self.create_optimization_task()
        ]

        crew = Crew(
            agents=[self.monitor_agent, self.optimizer_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=AGENT_CONFIG["verbose"]
        )

        result = _kickoff_with_retry(crew)
        return result

    def run_full_workflow(self, user_input: str) -> dict:
        """
        Run the complete workflow: Intent → Config → Monitor → Optimize

        This demonstrates the full 5G-Advanced AI-native optimization cycle.
        """
        tasks = [
            self.create_intent_task(user_input),
            self.create_planning_task(),
            self.create_monitoring_task(),
            self.create_optimization_task()
        ]

        crew = Crew(
            agents=[
                self.intent_agent,
                self.planner_agent,
                self.monitor_agent,
                self.optimizer_agent
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=AGENT_CONFIG["verbose"]
        )

        result = _kickoff_with_retry(crew)
        return result

    # =========================================================================
    # App-facing helpers — route UI pipeline steps through Groq LLM agents
    # =========================================================================

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    @staticmethod
    def _kickoff(crew):
        """Run a crew with automatic retry on Groq rate-limit errors."""
        return _kickoff_with_retry(crew)

    @staticmethod
    def _parse_raw(result) -> dict:
        """Strip markdown fences and JSON-parse the crew output."""
        raw = re.sub(r"```(?:json)?", "", str(result)).strip().rstrip("`").strip()
        return json.loads(raw)

    # ------------------------------------------------------------------
    # App-facing helpers — route UI pipeline steps through Groq LLM agents
    # ------------------------------------------------------------------

    def run_intent_for_app(self, user_intent: str) -> dict:
        """
        Run the intent agent (Groq LLM) to parse natural language into structured intent.
        Falls back to _parse_intent_impl if output cannot be parsed.
        """
        task = Task(
            description=(
                f'Parse this 5G network intent: "{user_intent}"\n'
                "Use parse_intent tool. Return ONLY the raw JSON result."
            ),
            expected_output="JSON intent dict from parse_intent tool.",
            agent=self.intent_agent
        )
        crew = Crew(agents=[self.intent_agent], tasks=[task],
                    process=Process.sequential, verbose=AGENT_CONFIG["verbose"])
        try:
            return self._parse_raw(self._kickoff(crew))
        except Exception:
            from tools.intent_tools import _parse_intent_impl
            return _parse_intent_impl(user_intent)

    def run_reasoner_for_app(self, intent_result: dict, answers: dict, metrics: dict):
        """
        Run the planner agent (Groq LLM) to analyse feasibility, risks, and impact.
        Returns {"feasibility":{...}, "risks":[...], "impact":{...}} or None on failure.
        Prompt is kept compact to stay within the free-tier TPM budget.
        """
        # Extract only the fields the LLM really needs (keep prompt small)
        m = metrics  # already a flat dict of KPI values
        intent_summary = {
            "intent_type": intent_result.get("intent_type"),
            "slice_type": intent_result.get("slice_type"),
            "entities": intent_result.get("entities", {}),
        }
        task = Task(
            description=(
                f"Intent: {json.dumps(intent_summary)}\n"
                f"Answers: {json.dumps(answers)}\n"
                f"Metrics: latency={m.get('latency_ms')}ms, "
                f"throughput={m.get('throughput_mbps')}Mbps, "
                f"cell_load={m.get('cell_load_percent')}%, "
                f"users={m.get('connected_users')}\n\n"
                "Return ONLY this JSON (no markdown):\n"
                '{"feasibility":{"feasible":true,"feasibility_score":<0-100>,"constraints":[]},'
                '"risks":[{"risk":"<name>","severity":"<HIGH|MEDIUM|LOW>","description":"<text>","mitigation":"<text>"}],'
                '"impact":{"before":{"latency_ms":<f>,"throughput_mbps":<f>,"cell_load_percent":<f>,"connected_users":<i>},'
                '"predicted_after":{"latency_ms":<f>,"throughput_mbps":<f>,"cell_load_percent":<f>,"connected_users":<i>},'
                '"summary":"<text>"}}'
            ),
            expected_output="JSON with feasibility, risks, and impact.",
            agent=self.planner_agent
        )
        crew = Crew(agents=[self.planner_agent], tasks=[task],
                    process=Process.sequential, verbose=AGENT_CONFIG["verbose"])
        try:
            return self._parse_raw(self._kickoff(crew))
        except Exception:
            return None  # caller falls back to rule-based functions

    def run_planner_for_app(self, intent_result: dict) -> dict:
        """
        Run the planner agent (Groq LLM) to generate a network configuration.
        Falls back to _generate_config_impl if output cannot be parsed.
        """
        intent_summary = {
            "intent_type": intent_result.get("intent_type"),
            "slice_type": intent_result.get("slice_type"),
            "entities": intent_result.get("entities", {}),
        }
        task = Task(
            description=(
                f"Intent: {json.dumps(intent_summary)}\n"
                "Use generate_config tool. Return ONLY the raw JSON config result."
            ),
            expected_output="JSON config dict from generate_config tool.",
            agent=self.planner_agent
        )
        crew = Crew(agents=[self.planner_agent], tasks=[task],
                    process=Process.sequential, verbose=AGENT_CONFIG["verbose"])
        try:
            return self._parse_raw(self._kickoff(crew))
        except Exception:
            from tools.config_tools import _generate_config_impl
            return _generate_config_impl(intent_result)

    def run_monitoring_for_app(self):
        """
        Run the monitor agent (Groq LLM) to assess network health.
        Returns (metrics_result, status_result). Falls back to direct tool calls.
        """
        from tools.monitor_tools import _get_metrics_impl, _check_status_impl
        metrics_result = _get_metrics_impl()
        m = metrics_result.get("metrics", {})

        task = Task(
            description=(
                f"Live KPIs: latency={m.get('latency_ms')}ms, "
                f"throughput={m.get('throughput_mbps')}Mbps, "
                f"cell_load={m.get('cell_load_percent')}%, "
                f"packet_loss={m.get('packet_loss_percent')}%, "
                f"users={m.get('connected_users')}\n"
                "Use check_status tool. Return ONLY the raw JSON status result."
            ),
            expected_output="JSON status dict from check_status tool.",
            agent=self.monitor_agent
        )
        crew = Crew(agents=[self.monitor_agent], tasks=[task],
                    process=Process.sequential, verbose=AGENT_CONFIG["verbose"])
        try:
            status_result = self._parse_raw(self._kickoff(crew))
            return metrics_result, status_result
        except Exception:
            return metrics_result, _check_status_impl(metrics_result)

    def run_optimizer_for_app(self, status_result: dict) -> dict:
        """
        Run the optimizer agent (Groq LLM) to select and execute the best RAN action.
        Falls back to rule-based selection if output cannot be parsed.
        """
        # Trim to the fields the agent really needs
        status_summary = {
            "overall_status": status_result.get("overall_status"),
            "health_score": status_result.get("health_score"),
            "violations": status_result.get("violations", []),
            "recommendations": status_result.get("recommendations", [])[:3],
        }
        task = Task(
            description=(
                f"Network status: {json.dumps(status_summary)}\n"
                "Choose the best action (scale_bandwidth|activate_cell|adjust_priority|modify_qos|energy_saving), "
                "set parameters, call execute_action. Return ONLY the raw JSON result."
            ),
            expected_output="JSON optimization result from execute_action tool.",
            agent=self.optimizer_agent
        )
        crew = Crew(agents=[self.optimizer_agent], tasks=[task],
                    process=Process.sequential, verbose=AGENT_CONFIG["verbose"])
        try:
            return self._parse_raw(self._kickoff(crew))
        except Exception:
            recs = status_result.get("recommendations", [])
            action = recs[0].get("action", "scale_bandwidth") if recs else "scale_bandwidth"
            params = {"change_mbps": 50} if action == "scale_bandwidth" else {"count": 2}
            from tools.action_tools import _execute_action_impl
            return _execute_action_impl(action, params)
