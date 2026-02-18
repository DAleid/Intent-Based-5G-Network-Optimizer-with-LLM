"""
Monitoring Tools for 5G-Advanced Network Optimizer

Tools used by the Monitor & Analyzer Agent to track network performance.
"""

import json
import re
import sys
from typing import Dict, Any, List
from crewai.tools import tool
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.network_sim import network_simulator
from config.settings import KPI_THRESHOLDS


def _get_metrics_impl() -> Dict[str, Any]:
    """
    Internal implementation: Collects current network performance metrics.
    """
    metrics = network_simulator.get_metrics()

    return {
        "timestamp": metrics.timestamp.isoformat(),
        "metrics": {
            "throughput_mbps": metrics.throughput_mbps,
            "latency_ms": metrics.latency_ms,
            "packet_loss_percent": metrics.packet_loss_percent,
            "connected_users": metrics.connected_users,
            "cell_load_percent": metrics.cell_load_percent,
            "active_slices": metrics.active_slices,
            "energy_consumption_kwh": metrics.energy_consumption_kwh
        },
        "collection_successful": True
    }


@tool("Get Metrics")
def get_metrics() -> Dict[str, Any]:
    """
    Collects current network performance metrics from the 5G-Advanced network.

    Retrieves all key performance indicators (KPIs):
    - Throughput (Mbps)
    - Latency (ms)
    - Packet Loss (%)
    - Connected Users
    - Cell Load (%)
    - Active Slices
    - Energy Consumption (kWh)

    Returns:
        Dictionary containing all current network metrics with timestamp
    """
    return _get_metrics_impl()


def _check_status_impl(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Internal implementation: Analyzes network metrics and detects issues."""
    # Handle both direct metrics and wrapped metrics
    if "metrics" in metrics:
        metric_values = metrics["metrics"]
    else:
        metric_values = metrics

    # Step 1: Check thresholds
    violations = _check_thresholds(metric_values)

    # Step 2: Detect anomalies
    anomalies = _detect_anomalies(metric_values)

    # Step 3: Analyze trends
    trend_analysis = _analyze_trends(metric_values)

    # Step 4: Determine overall status
    overall_status = _determine_overall_status(violations, anomalies)

    # Step 5: Generate recommendations
    recommendations = _generate_recommendations(violations, anomalies, trend_analysis)

    return {
        "analysis_timestamp": metrics.get("timestamp", "N/A"),
        "overall_status": overall_status,
        "health_score": _calculate_health_score(violations, anomalies),
        "violations": violations,
        "anomalies": anomalies,
        "trend_analysis": trend_analysis,
        "predictions": _generate_predictions(metric_values, trend_analysis),
        "recommendations": recommendations,
        "requires_action": overall_status in ["warning", "critical"],
        "steps_executed": [
            "threshold_check",
            "anomaly_detection",
            "trend_analysis",
            "alert_generation"
        ]
    }


@tool("Check Status")
def check_status(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes network metrics and detects issues, anomalies, and trends.

    This tool performs comprehensive analysis:
    1. Threshold checking - compares metrics against defined limits
    2. Anomaly detection - identifies unusual patterns
    3. Trend analysis - predicts future metric values
    4. Alert generation - creates actionable alerts

    Args:
        metrics: Current network metrics (from get_metrics tool)

    Returns:
        Detailed analysis including status, violations, predictions, and recommendations
    """
    return _check_status_impl(metrics)


def _check_thresholds(metrics: Dict) -> List[Dict]:
    """Check metrics against defined thresholds"""

    violations = []

    # Latency check
    latency = metrics.get("latency_ms", 0)
    latency_thresholds = KPI_THRESHOLDS["latency_ms"]

    if latency >= latency_thresholds["critical"]:
        violations.append({
            "metric": "latency_ms",
            "current_value": latency,
            "threshold": latency_thresholds["critical"],
            "severity": "critical",
            "message": f"Latency critically high: {latency}ms (threshold: {latency_thresholds['critical']}ms)"
        })
    elif latency >= latency_thresholds["warning"]:
        violations.append({
            "metric": "latency_ms",
            "current_value": latency,
            "threshold": latency_thresholds["warning"],
            "severity": "warning",
            "message": f"Latency elevated: {latency}ms (threshold: {latency_thresholds['warning']}ms)"
        })

    # Throughput check (lower is worse)
    throughput = metrics.get("throughput_mbps", 100)
    throughput_thresholds = KPI_THRESHOLDS["throughput_mbps"]

    if throughput <= throughput_thresholds["critical"]:
        violations.append({
            "metric": "throughput_mbps",
            "current_value": throughput,
            "threshold": throughput_thresholds["critical"],
            "severity": "critical",
            "message": f"Throughput critically low: {throughput}Mbps (minimum: {throughput_thresholds['critical']}Mbps)"
        })
    elif throughput <= throughput_thresholds["warning"]:
        violations.append({
            "metric": "throughput_mbps",
            "current_value": throughput,
            "threshold": throughput_thresholds["warning"],
            "severity": "warning",
            "message": f"Throughput below target: {throughput}Mbps (target: {throughput_thresholds['warning']}Mbps)"
        })

    # Cell load check
    cell_load = metrics.get("cell_load_percent", 50)
    cell_load_thresholds = KPI_THRESHOLDS["cell_load_percent"]

    if cell_load >= cell_load_thresholds["critical"]:
        violations.append({
            "metric": "cell_load_percent",
            "current_value": cell_load,
            "threshold": cell_load_thresholds["critical"],
            "severity": "critical",
            "message": f"Cell load critical: {cell_load}% (threshold: {cell_load_thresholds['critical']}%)"
        })
    elif cell_load >= cell_load_thresholds["warning"]:
        violations.append({
            "metric": "cell_load_percent",
            "current_value": cell_load,
            "threshold": cell_load_thresholds["warning"],
            "severity": "warning",
            "message": f"Cell load high: {cell_load}% (threshold: {cell_load_thresholds['warning']}%)"
        })

    # Packet loss check
    packet_loss = metrics.get("packet_loss_percent", 0)
    packet_loss_thresholds = KPI_THRESHOLDS["packet_loss_percent"]

    if packet_loss >= packet_loss_thresholds["critical"]:
        violations.append({
            "metric": "packet_loss_percent",
            "current_value": packet_loss,
            "threshold": packet_loss_thresholds["critical"],
            "severity": "critical",
            "message": f"Packet loss critical: {packet_loss}% (threshold: {packet_loss_thresholds['critical']}%)"
        })
    elif packet_loss >= packet_loss_thresholds["warning"]:
        violations.append({
            "metric": "packet_loss_percent",
            "current_value": packet_loss,
            "threshold": packet_loss_thresholds["warning"],
            "severity": "warning",
            "message": f"Packet loss elevated: {packet_loss}% (threshold: {packet_loss_thresholds['warning']}%)"
        })

    return violations


def _detect_anomalies(metrics: Dict) -> List[Dict]:
    """Detect anomalies in metrics"""

    anomalies = []

    # Check for unusual combinations
    latency = metrics.get("latency_ms", 0)
    throughput = metrics.get("throughput_mbps", 100)
    cell_load = metrics.get("cell_load_percent", 50)

    # High latency with low cell load is anomalous
    if latency > 60 and cell_load < 50:
        anomalies.append({
            "type": "latency_cell_load_mismatch",
            "description": "High latency detected despite low cell load",
            "severity": "medium",
            "possible_cause": "Possible backhaul issue or configuration problem"
        })

    # Low throughput with low cell load is anomalous
    if throughput < 70 and cell_load < 40:
        anomalies.append({
            "type": "throughput_capacity_mismatch",
            "description": "Low throughput despite available capacity",
            "severity": "medium",
            "possible_cause": "Possible interference or spectrum issue"
        })

    # Sudden spikes (simulated based on thresholds)
    if latency > KPI_THRESHOLDS["latency_ms"]["warning"] * 1.5:
        anomalies.append({
            "type": "latency_spike",
            "description": "Sudden latency spike detected",
            "severity": "high",
            "possible_cause": "Traffic surge or network congestion"
        })

    return anomalies


def _analyze_trends(metrics: Dict) -> Dict:
    """Analyze metric trends"""

    # In a real system, this would use historical data
    # For the prototype, we simulate trend analysis

    latency = metrics.get("latency_ms", 20)
    throughput = metrics.get("throughput_mbps", 100)
    cell_load = metrics.get("cell_load_percent", 50)

    # Simulate trend based on current values relative to thresholds
    latency_trend = "increasing" if latency > 50 else "stable"
    throughput_trend = "decreasing" if throughput < 80 else "stable"
    cell_load_trend = "increasing" if cell_load > 60 else "stable"

    return {
        "latency": {
            "trend": latency_trend,
            "direction": "up" if latency_trend == "increasing" else "stable",
            "rate_of_change": "moderate" if latency_trend == "increasing" else "none"
        },
        "throughput": {
            "trend": throughput_trend,
            "direction": "down" if throughput_trend == "decreasing" else "stable",
            "rate_of_change": "slow" if throughput_trend == "decreasing" else "none"
        },
        "cell_load": {
            "trend": cell_load_trend,
            "direction": "up" if cell_load_trend == "increasing" else "stable",
            "rate_of_change": "moderate" if cell_load_trend == "increasing" else "none"
        }
    }


def _generate_predictions(metrics: Dict, trends: Dict) -> Dict:
    """Generate predictions based on current metrics and trends"""

    latency = metrics.get("latency_ms", 20)
    cell_load = metrics.get("cell_load_percent", 50)

    predictions = {}

    # Latency prediction
    if trends["latency"]["trend"] == "increasing":
        predicted_latency = latency * 1.3
        time_to_threshold = _estimate_time_to_threshold(
            latency,
            KPI_THRESHOLDS["latency_ms"]["critical"],
            "increasing"
        )
        predictions["latency"] = {
            "predicted_value_5min": round(predicted_latency, 1),
            "will_exceed_threshold": predicted_latency > KPI_THRESHOLDS["latency_ms"]["warning"],
            "time_to_threshold_minutes": time_to_threshold
        }

    # Cell load prediction
    if trends["cell_load"]["trend"] == "increasing":
        predicted_load = cell_load * 1.2
        predictions["cell_load"] = {
            "predicted_value_5min": round(predicted_load, 1),
            "will_exceed_threshold": predicted_load > KPI_THRESHOLDS["cell_load_percent"]["warning"],
            "time_to_threshold_minutes": _estimate_time_to_threshold(
                cell_load,
                KPI_THRESHOLDS["cell_load_percent"]["critical"],
                "increasing"
            )
        }

    return predictions


def _estimate_time_to_threshold(current: float, threshold: float, direction: str) -> int:
    """Estimate minutes until threshold is reached"""

    if direction != "increasing" or current >= threshold:
        return -1  # Not approaching threshold

    # Simple linear estimation
    rate = 0.1  # 10% increase per minute (simulated)
    if rate <= 0:
        return -1

    minutes = 0
    value = current
    while value < threshold and minutes < 60:
        value *= (1 + rate)
        minutes += 1

    return minutes if minutes < 60 else -1


def _determine_overall_status(violations: List, anomalies: List) -> str:
    """Determine overall network status"""

    critical_violations = [v for v in violations if v["severity"] == "critical"]
    warning_violations = [v for v in violations if v["severity"] == "warning"]
    high_anomalies = [a for a in anomalies if a["severity"] == "high"]

    if critical_violations or high_anomalies:
        return "critical"
    elif warning_violations or anomalies:
        return "warning"
    else:
        return "healthy"


def _calculate_health_score(violations: List, anomalies: List) -> int:
    """Calculate health score from 0-100"""

    score = 100

    for v in violations:
        if v["severity"] == "critical":
            score -= 30
        elif v["severity"] == "warning":
            score -= 15

    for a in anomalies:
        if a["severity"] == "high":
            score -= 20
        elif a["severity"] == "medium":
            score -= 10

    return max(0, score)


def _generate_recommendations(
    violations: List,
    anomalies: List,
    trends: Dict
) -> List[Dict]:
    """Generate actionable recommendations"""

    recommendations = []

    for violation in violations:
        metric = violation["metric"]

        if metric == "latency_ms":
            recommendations.append({
                "action": "scale_bandwidth",
                "priority": "high" if violation["severity"] == "critical" else "medium",
                "reason": violation["message"],
                "expected_improvement": "20-40% latency reduction"
            })

        elif metric == "cell_load_percent":
            recommendations.append({
                "action": "activate_cell",
                "priority": "high" if violation["severity"] == "critical" else "medium",
                "reason": violation["message"],
                "expected_improvement": "15-30% load reduction"
            })

        elif metric == "throughput_mbps":
            recommendations.append({
                "action": "scale_bandwidth",
                "priority": "medium",
                "reason": violation["message"],
                "expected_improvement": "Increased throughput capacity"
            })

    # Add recommendations based on trends
    if trends.get("latency", {}).get("trend") == "increasing":
        recommendations.append({
            "action": "modify_qos",
            "priority": "low",
            "reason": "Latency trending upward",
            "expected_improvement": "Preventive QoS optimization"
        })

    return recommendations
