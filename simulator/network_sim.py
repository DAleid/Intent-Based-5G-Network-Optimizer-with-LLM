"""
5G-Advanced Network Simulator

Simulates a 5G-Advanced network with:
- Network Slicing (eMBB, URLLC, mMTC)
- Dynamic KPI generation from REAL dataset
- Real-time optimization responses
- 3GPP Release 18 compliant parameters

Data Source: 6G HetNet Transmission Management Dataset
"""

import random
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class NetworkSlice:
    """Represents a 5G-Advanced Network Slice"""
    slice_id: str
    slice_type: str  # eMBB, URLLC, mMTC
    sst: int         # Slice/Service Type (1=eMBB, 2=URLLC, 3=mMTC)
    allocated_bandwidth_mbps: float
    latency_target_ms: float
    priority: int
    active: bool = True
    current_users: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkMetrics:
    """Current network performance metrics from 6G HetNet Dataset"""
    timestamp: datetime
    throughput_mbps: float
    latency_ms: float
    packet_loss_percent: float
    connected_users: int
    cell_load_percent: float
    active_slices: int
    energy_consumption_kwh: float
    # Extended metrics from real dataset
    cell_type: str = "Macro"                    # Macro, Micro, Pico, Femto
    carrier_frequency_ghz: float = 3.5          # Carrier frequency
    bandwidth_mhz: float = 100.0                # Bandwidth
    modulation_scheme: str = "64QAM"            # Modulation type
    transmission_power_dbm: float = 30.0        # TX power
    interference_level_db: float = -100.0       # Interference
    snr_db: float = 20.0                        # Signal-to-Noise Ratio
    energy_efficiency_mbps_watt: float = 1.0    # Energy efficiency
    user_traffic_demand_mbps: float = 50.0      # User demand
    qos_satisfaction: float = 0.9               # QoS score (0-1)
    user_mobility_kmh: float = 0.0              # User speed
    handover_count: int = 0                     # Handovers


class Network5GAdvancedSimulator:
    """
    Simulates a 5G-Advanced network for the Agentic AI optimizer.

    Features:
    - Dynamic KPI generation from REAL 6G HetNet dataset
    - Network slice management
    - Simulated optimization actions
    - Event-based traffic patterns (stadium, emergency, etc.)
    """

    def __init__(self):
        self.slices: Dict[str, NetworkSlice] = {}

        # Load real dataset
        self._load_dataset()

        self.base_metrics = {
            "throughput_mbps": 100.0,
            "latency_ms": 20.0,
            "packet_loss_percent": 0.01,
            "connected_users": 1000,
            "cell_load_percent": 50.0,
            "active_cells": 20,
            "energy_consumption_kwh": 50.0
        }
        self.current_metrics = self.base_metrics.copy()
        self.optimization_history: List[Dict] = []
        self.event_active = False
        self.event_type = None
        self._anomaly_active = False
        self._anomaly_start_time = None

    def _load_dataset(self):
        """Load the 6G HetNet Transmission Management dataset"""
        try:
            # Find the dataset file
            data_path = Path(__file__).parent.parent / "data" / "6G_HetNet_Transmission_Management.csv"

            if data_path.exists():
                self.dataset = pd.read_csv(data_path)
                self._dataset_index = 0
                self._use_real_data = True
                print(f"[OK] Loaded real dataset with {len(self.dataset)} records")
            else:
                self._use_real_data = False
                self.dataset = None
                print("[WARNING] Dataset not found, using simulated data")
        except Exception as e:
            self._use_real_data = False
            self.dataset = None
            print(f"[WARNING] Error loading dataset: {e}, using simulated data")

    def create_slice(
        self,
        slice_type: str,
        bandwidth_mbps: float,
        latency_target_ms: float,
        priority: int = 5
    ) -> NetworkSlice:
        """
        Create a new network slice (5G-Advanced feature).

        Args:
            slice_type: eMBB, URLLC, or mMTC
            bandwidth_mbps: Allocated bandwidth
            latency_target_ms: Target latency
            priority: 1 (highest) to 10 (lowest)

        Returns:
            Created NetworkSlice object
        """
        sst_mapping = {"eMBB": 1, "URLLC": 2, "mMTC": 3}
        slice_id = f"slice_{slice_type}_{int(time.time())}"

        new_slice = NetworkSlice(
            slice_id=slice_id,
            slice_type=slice_type,
            sst=sst_mapping.get(slice_type, 1),
            allocated_bandwidth_mbps=bandwidth_mbps,
            latency_target_ms=latency_target_ms,
            priority=priority
        )

        self.slices[slice_id] = new_slice
        self._update_metrics_for_slice(new_slice)

        return new_slice

    def _update_metrics_for_slice(self, slice: NetworkSlice):
        """Update network metrics when a slice is created"""
        # Adding a slice improves overall performance initially
        self.current_metrics["throughput_mbps"] += slice.allocated_bandwidth_mbps * 0.8
        self.current_metrics["active_cells"] += 2

    def get_metrics(self) -> NetworkMetrics:
        """
        Get current network metrics from REAL dataset with realistic variations.

        Returns:
            NetworkMetrics object with current values
        """
        # Use real dataset if available
        if self._use_real_data and self.dataset is not None:
            return self._get_metrics_from_dataset()

        # Fallback to simulated data
        return self._get_simulated_metrics()

    def _get_metrics_from_dataset(self) -> NetworkMetrics:
        """Get metrics from the real 6G HetNet dataset - ALL COLUMNS"""
        # Get current row from dataset (cycle through)
        row = self.dataset.iloc[self._dataset_index]

        # Move to next row, wrap around if needed
        self._dataset_index = (self._dataset_index + 1) % len(self.dataset)

        # === CORE METRICS ===
        throughput = float(row.get('Achieved_Throughput_Mbps', 100.0))
        latency = float(row.get('Network_Latency_ms', 20.0))
        packet_loss = float(row.get('Packet_Loss_Ratio', 0.01)) * 100  # Convert ratio to percent
        cell_load = float(row.get('Resource_Utilization', 50.0)) * 100  # Convert to percent
        energy = float(row.get('Power_Consumption_Watt', 50.0)) / 1000  # Convert W to kWh estimate

        # === EXTENDED METRICS FROM DATASET ===
        cell_type = str(row.get('Cell_Type', 'Macro'))
        carrier_frequency = float(row.get('Carrier_Frequency_GHz', 3.5))
        bandwidth = float(row.get('Bandwidth_MHz', 100.0))
        modulation = str(row.get('Modulation_Scheme', '64QAM'))
        tx_power = float(row.get('Transmission_Power_dBm', 30.0))
        interference = float(row.get('Interference_Level_dB', -100.0))
        snr = float(row.get('Signal_to_Noise_Ratio_dB', 20.0))
        energy_efficiency = float(row.get('Energy_Efficiency_Mbps_Watt', 1.0))
        traffic_demand = float(row.get('User_Traffic_Demand_Mbps', 50.0))
        qos = float(row.get('QoS_Satisfaction', 0.9))
        mobility = float(row.get('User_Mobility_kmh', 0.0))
        handovers = int(row.get('Handover_Count', 0))

        # Estimate connected users based on resource utilization and cell type
        cell_capacity = {'Macro': 1000, 'Micro': 500, 'Pico': 200, 'Femto': 50}
        base_capacity = cell_capacity.get(cell_type, 500)
        users = int(base_capacity * (cell_load / 100))

        # Apply event effects if active
        if self.event_active:
            throughput, latency, users, cell_load = self._apply_event_effects(
                throughput, latency, users, cell_load
            )

        # Apply anomaly effects
        if self._anomaly_active:
            anomaly_factor = self._calculate_anomaly_factor()
            throughput /= anomaly_factor
            latency *= anomaly_factor
            packet_loss *= anomaly_factor
            cell_load = min(100, cell_load * anomaly_factor)
            qos /= anomaly_factor  # QoS degrades during anomaly
        elif random.random() < 0.05:  # Lower chance with real data
            self._start_anomaly()

        return NetworkMetrics(
            timestamp=datetime.now(),
            throughput_mbps=round(throughput, 2),
            latency_ms=round(latency, 2),
            packet_loss_percent=round(packet_loss, 4),
            connected_users=users,
            cell_load_percent=round(min(100, cell_load), 2),
            active_slices=len([s for s in self.slices.values() if s.active]),
            energy_consumption_kwh=round(energy, 2),
            # Extended metrics
            cell_type=cell_type,
            carrier_frequency_ghz=round(carrier_frequency, 2),
            bandwidth_mhz=round(bandwidth, 2),
            modulation_scheme=modulation,
            transmission_power_dbm=round(tx_power, 2),
            interference_level_db=round(interference, 2),
            snr_db=round(snr, 2),
            energy_efficiency_mbps_watt=round(energy_efficiency, 2),
            user_traffic_demand_mbps=round(traffic_demand, 2),
            qos_satisfaction=round(min(1.0, qos), 4),
            user_mobility_kmh=round(mobility, 2),
            handover_count=handovers
        )

    def _get_simulated_metrics(self) -> NetworkMetrics:
        """Fallback simulated metrics generation"""
        # Add realistic variations
        variation = random.uniform(0.95, 1.05)

        # Check for anomaly simulation
        if self._anomaly_active:
            anomaly_factor = self._calculate_anomaly_factor()
        else:
            anomaly_factor = 1.0
            # Random chance of anomaly starting
            if random.random() < 0.1:
                self._start_anomaly()

        # Calculate current metrics
        throughput = self.current_metrics["throughput_mbps"] * variation / anomaly_factor
        latency = self.current_metrics["latency_ms"] * variation * anomaly_factor
        packet_loss = self.current_metrics["packet_loss_percent"] * variation * anomaly_factor
        users = int(self.current_metrics["connected_users"] * variation)
        cell_load = min(100, self.current_metrics["cell_load_percent"] * variation * anomaly_factor)

        # Event-based modifications
        if self.event_active:
            throughput, latency, users, cell_load = self._apply_event_effects(
                throughput, latency, users, cell_load
            )

        return NetworkMetrics(
            timestamp=datetime.now(),
            throughput_mbps=round(throughput, 2),
            latency_ms=round(latency, 2),
            packet_loss_percent=round(packet_loss, 4),
            connected_users=users,
            cell_load_percent=round(cell_load, 2),
            active_slices=len([s for s in self.slices.values() if s.active]),
            energy_consumption_kwh=round(self.current_metrics["energy_consumption_kwh"] * variation, 2)
        )

    def _start_anomaly(self):
        """Start a simulated network anomaly"""
        self._anomaly_active = True
        self._anomaly_start_time = time.time()

    def _calculate_anomaly_factor(self) -> float:
        """Calculate how severe the current anomaly is"""
        if not self._anomaly_active:
            return 1.0

        elapsed = time.time() - self._anomaly_start_time

        # Anomaly grows for 30 seconds then resolves
        if elapsed > 30:
            self._anomaly_active = False
            return 1.0

        # Peak anomaly at 15 seconds
        if elapsed < 15:
            return 1.0 + (elapsed / 15) * 0.8  # Up to 1.8x degradation
        else:
            return 1.8 - ((elapsed - 15) / 15) * 0.8  # Recovering

    def _apply_event_effects(
        self,
        throughput: float,
        latency: float,
        users: int,
        cell_load: float
    ) -> tuple:
        """Apply event-specific effects to metrics"""

        event_effects = {
            "stadium": {"users_mult": 5.0, "load_add": 30, "latency_mult": 1.5},
            "emergency": {"users_mult": 1.2, "load_add": 10, "latency_mult": 0.5},
            "iot_deployment": {"users_mult": 10.0, "load_add": 20, "latency_mult": 1.2},
            "concert": {"users_mult": 4.0, "load_add": 25, "latency_mult": 1.4},
            "healthcare": {"users_mult": 1.5, "load_add": 15, "latency_mult": 0.6},
            "transportation": {"users_mult": 2.0, "load_add": 25, "latency_mult": 0.4},
            "smart_factory": {"users_mult": 3.0, "load_add": 20, "latency_mult": 0.5},
            "video_conferencing": {"users_mult": 2.5, "load_add": 20, "latency_mult": 1.3},
            "gaming": {"users_mult": 3.5, "load_add": 25, "latency_mult": 0.7},
        }

        effects = event_effects.get(self.event_type, {})

        users = int(users * effects.get("users_mult", 1.0))
        cell_load = min(100, cell_load + effects.get("load_add", 0))
        latency = latency * effects.get("latency_mult", 1.0)

        return throughput, latency, users, cell_load

    def start_event(self, event_type: str):
        """Start a network event simulation"""
        self.event_active = True
        self.event_type = event_type

    def stop_event(self):
        """Stop the current event simulation"""
        self.event_active = False
        self.event_type = None

    def execute_optimization(self, action: str, parameters: Dict) -> Dict:
        """
        Execute an optimization action on the network.

        Args:
            action: Type of action (scale_bandwidth, activate_cell, etc.)
            parameters: Action-specific parameters

        Returns:
            Result of the optimization action
        """
        before_metrics = self.get_metrics()
        success = True
        message = ""

        if action == "scale_bandwidth":
            change = parameters.get("change_mbps", 50)
            self.current_metrics["throughput_mbps"] += change
            self.current_metrics["latency_ms"] *= 0.8  # Latency improves
            message = f"Bandwidth scaled by {change} Mbps"

        elif action == "activate_cell":
            count = parameters.get("count", 1)
            self.current_metrics["active_cells"] += count
            self.current_metrics["cell_load_percent"] *= 0.85  # Load distributed
            message = f"Activated {count} additional cell(s)"

        elif action == "adjust_priority":
            slice_id = parameters.get("slice_id")
            new_priority = parameters.get("priority", 1)
            if slice_id in self.slices:
                self.slices[slice_id].priority = new_priority
                self.current_metrics["latency_ms"] *= 0.7  # Priority traffic faster
                message = f"Adjusted priority for {slice_id} to {new_priority}"
            else:
                success = False
                message = f"Slice {slice_id} not found"

        elif action == "modify_qos":
            latency_target = parameters.get("latency_target_ms")
            if latency_target:
                self.current_metrics["latency_ms"] = min(
                    self.current_metrics["latency_ms"],
                    latency_target * 0.9
                )
                message = f"QoS modified, targeting {latency_target}ms latency"

        elif action == "energy_saving":
            mode = parameters.get("mode", "moderate")
            if mode == "aggressive":
                self.current_metrics["energy_consumption_kwh"] *= 0.7
                self.current_metrics["throughput_mbps"] *= 0.9
            else:
                self.current_metrics["energy_consumption_kwh"] *= 0.85
            message = f"Energy saving mode: {mode}"

        else:
            success = False
            message = f"Unknown action: {action}"

        # Clear anomaly if optimization was successful
        if success and self._anomaly_active:
            self._anomaly_active = False

        after_metrics = self.get_metrics()

        result = {
            "action": action,
            "parameters": parameters,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "before": {
                "latency_ms": before_metrics.latency_ms,
                "throughput_mbps": before_metrics.throughput_mbps,
                "cell_load_percent": before_metrics.cell_load_percent
            },
            "after": {
                "latency_ms": after_metrics.latency_ms,
                "throughput_mbps": after_metrics.throughput_mbps,
                "cell_load_percent": after_metrics.cell_load_percent
            }
        }

        self.optimization_history.append(result)
        return result

    def get_status_summary(self) -> Dict:
        """Get a summary of the current network status"""
        metrics = self.get_metrics()

        return {
            "overall_health": self._calculate_health_score(metrics),
            "active_slices": len([s for s in self.slices.values() if s.active]),
            "total_slices": len(self.slices),
            "event_active": self.event_active,
            "event_type": self.event_type,
            "anomaly_detected": self._anomaly_active,
            "metrics": {
                "throughput_mbps": metrics.throughput_mbps,
                "latency_ms": metrics.latency_ms,
                "packet_loss_percent": metrics.packet_loss_percent,
                "connected_users": metrics.connected_users,
                "cell_load_percent": metrics.cell_load_percent
            },
            "optimization_count": len(self.optimization_history)
        }

    def _calculate_health_score(self, metrics: NetworkMetrics) -> str:
        """Calculate overall network health"""
        score = 100

        # Latency penalty
        if metrics.latency_ms > 80:
            score -= 30
        elif metrics.latency_ms > 50:
            score -= 15

        # Throughput penalty
        if metrics.throughput_mbps < 50:
            score -= 25
        elif metrics.throughput_mbps < 80:
            score -= 10

        # Cell load penalty
        if metrics.cell_load_percent > 90:
            score -= 25
        elif metrics.cell_load_percent > 80:
            score -= 10

        if score >= 80:
            return "healthy"
        elif score >= 60:
            return "warning"
        else:
            return "critical"

    def reset(self):
        """Reset the simulator to initial state"""
        self.slices = {}
        self.current_metrics = self.base_metrics.copy()
        self.optimization_history = []
        self.event_active = False
        self.event_type = None
        self._anomaly_active = False
        # Reset dataset index if using real data
        if self._use_real_data:
            self._dataset_index = 0

    def get_data_source_info(self) -> Dict:
        """Get information about the current data source"""
        if self._use_real_data and self.dataset is not None:
            return {
                "source": "6G HetNet Transmission Management Dataset",
                "type": "real",
                "total_records": len(self.dataset),
                "current_index": self._dataset_index,
                "columns": list(self.dataset.columns)
            }
        else:
            return {
                "source": "Simulated Data",
                "type": "simulated",
                "total_records": None,
                "current_index": None,
                "columns": None
            }


# Global simulator instance
network_simulator = Network5GAdvancedSimulator()
