"""
Run the Monitoring & Healing Cycle

This script demonstrates the self-healing capability of the 5G-Advanced
network optimizer. It monitors the network and automatically applies
optimizations when issues are detected.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulator.network_sim import network_simulator
from tools.monitor_tools import _get_metrics_impl as get_metrics, _check_status_impl as check_status
from tools.action_tools import _execute_action_impl as execute_action
import time


def run_healing_cycle(verbose: bool = True):
    """
    Run a single monitoring and healing cycle.

    This implements the self-healing loop:
    Monitor → Analyze → Decide → Act → Verify
    """

    print("\n" + "="*60)
    print("🏥 5G-Advanced Network Self-Healing Cycle")
    print("="*60)

    # Step 1: Collect Metrics
    print("\n📊 Step 1: Collecting Network Metrics...")
    metrics = get_metrics()

    if verbose:
        print(f"   Throughput:   {metrics['metrics']['throughput_mbps']:.1f} Mbps")
        print(f"   Latency:      {metrics['metrics']['latency_ms']:.1f} ms")
        print(f"   Cell Load:    {metrics['metrics']['cell_load_percent']:.1f}%")
        print(f"   Packet Loss:  {metrics['metrics']['packet_loss_percent']:.4f}%")
        print(f"   Users:        {metrics['metrics']['connected_users']}")

    # Step 2: Analyze Status
    print("\n🔍 Step 2: Analyzing Network Status...")
    status = check_status(metrics)

    overall = status['overall_status']
    health_score = status['health_score']

    status_icon = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}.get(overall, '⚪')
    print(f"   Overall Status: {status_icon} {overall.upper()}")
    print(f"   Health Score:   {health_score}/100")

    # Show violations
    if status['violations']:
        print("\n   ⚠️ Violations Detected:")
        for v in status['violations']:
            print(f"      - {v['message']}")

    # Show anomalies
    if status['anomalies']:
        print("\n   🔮 Anomalies Detected:")
        for a in status['anomalies']:
            print(f"      - {a['description']}")

    # Step 3: Execute Healing (if needed)
    if status['requires_action']:
        print("\n🔧 Step 3: Executing Self-Healing Actions...")

        for rec in status['recommendations']:
            action = rec['action']
            priority = rec['priority']
            reason = rec['reason']

            print(f"\n   Action: {action}")
            print(f"   Priority: {priority}")
            print(f"   Reason: {reason}")

            # Determine parameters based on action
            if action == "scale_bandwidth":
                params = {"change_mbps": 50}
            elif action == "activate_cell":
                params = {"count": 2}
            elif action == "adjust_priority":
                params = {"slice_id": "default", "priority": 1}
            elif action == "modify_qos":
                params = {"latency_target_ms": 30}
            else:
                params = {}

            # Execute the action
            result = execute_action(action, params)

            if result['success']:
                print(f"   ✅ Action Successful!")

                # Show improvements
                if 'execution_details' in result:
                    improvements = result['execution_details'].get('improvements', {})
                    if 'latency' in improvements:
                        print(f"      Latency: {improvements['latency']['description']}")
                    if 'throughput' in improvements:
                        print(f"      Throughput: {improvements['throughput']['description']}")
                    if 'cell_load' in improvements:
                        print(f"      Cell Load: {improvements['cell_load']['description']}")
            else:
                print(f"   ❌ Action Failed: {result.get('message', 'Unknown error')}")
    else:
        print("\n✅ Step 3: No healing needed - Network is healthy!")

    # Step 4: Verify
    print("\n📋 Step 4: Verifying Network Status...")
    final_metrics = get_metrics()
    final_status = check_status(final_metrics)

    final_icon = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}.get(final_status['overall_status'], '⚪')
    print(f"   Final Status: {final_icon} {final_status['overall_status'].upper()}")
    print(f"   Final Health: {final_status['health_score']}/100")

    print("\n" + "="*60)
    print("🏥 Healing Cycle Complete")
    print("="*60 + "\n")

    return final_status


def simulate_problem_and_heal():
    """
    Simulate a network problem and demonstrate self-healing.
    """
    print("\n" + "🎬 "*20)
    print("DEMO: Simulating Network Problem and Self-Healing")
    print("🎬 "*20)

    # Start with healthy network
    print("\n1️⃣ Initial State (Healthy Network):")
    network_simulator.reset()
    run_healing_cycle(verbose=False)

    # Simulate a stadium event (causes stress)
    print("\n2️⃣ Simulating Stadium Event (Network Stress)...")
    network_simulator.start_event("stadium")
    time.sleep(1)

    # Force an anomaly for demonstration
    network_simulator._anomaly_active = True
    network_simulator._anomaly_start_time = time.time() - 10  # 10 seconds into anomaly

    print("\n3️⃣ Running Healing Cycle (Problem Detection & Resolution):")
    run_healing_cycle(verbose=True)

    # Stop the event
    print("\n4️⃣ Event Ended - Final Verification:")
    network_simulator.stop_event()
    run_healing_cycle(verbose=False)


def continuous_monitoring(interval_seconds: int = 10, max_cycles: int = 5):
    """
    Run continuous monitoring and healing.

    Args:
        interval_seconds: Time between monitoring cycles
        max_cycles: Maximum number of cycles to run (0 for infinite)
    """
    print("\n" + "="*60)
    print("🔄 Starting Continuous Monitoring Mode")
    print(f"   Interval: {interval_seconds} seconds")
    print(f"   Max Cycles: {max_cycles if max_cycles > 0 else 'Infinite'}")
    print("   Press Ctrl+C to stop")
    print("="*60)

    cycle = 0
    try:
        while max_cycles == 0 or cycle < max_cycles:
            cycle += 1
            print(f"\n--- Cycle {cycle} ---")
            run_healing_cycle(verbose=True)

            if max_cycles == 0 or cycle < max_cycles:
                print(f"\n⏳ Waiting {interval_seconds} seconds...")
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="5G-Advanced Network Self-Healing")
    parser.add_argument("--mode", choices=["single", "demo", "continuous"],
                        default="demo", help="Running mode")
    parser.add_argument("--interval", type=int, default=10,
                        help="Interval for continuous mode (seconds)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Max cycles for continuous mode (0=infinite)")

    args = parser.parse_args()

    if args.mode == "single":
        run_healing_cycle()
    elif args.mode == "demo":
        simulate_problem_and_heal()
    elif args.mode == "continuous":
        continuous_monitoring(args.interval, args.cycles)
