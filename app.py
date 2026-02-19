"""
5G-Advanced Network Optimizer - Streamlit Application

Agentic AI for Real-Time Optimization of 5G-Advanced Networks

This application provides a visual interface for:
- Entering network optimization intents
- Viewing agent activity in real-time
- Monitoring network KPIs
- Observing autonomous optimizations
"""

import streamlit as st
import time
import json
import random
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simulator.network_sim import network_simulator
from tools.config_tools import _generate_config_impl as generate_config
from tools.monitor_tools import _get_metrics_impl as get_metrics, _check_status_impl as check_status
from tools.action_tools import _execute_action_impl as execute_action
from config.settings import KPI_THRESHOLDS, NETWORK_CONFIG
from tools.intent_tools import _parse_intent_impl as parse_intent
from tools.reasoning_llm import (
    generate_reasoning_questions as _llm_questions,
    resolve_conflicts_with_llm,
    map_intent_to_cells_with_llm,
)
from agents.crew import Network5GOptimizationCrew


@st.cache_resource
def get_optimization_crew():
    """Instantiate the Groq-powered agent crew once and cache it for the session."""
    return Network5GOptimizationCrew()

# Page configuration
st.set_page_config(
    page_title="Autonomous Intent-Based 5G-Advanced Network Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        color: #212529 !important;
    }
    .status-healthy { color: #198754 !important; }
    .status-warning { color: #ffc107 !important; }
    .status-critical { color: #dc3545 !important; }
    .stButton > button {
        width: 100%;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .alert-pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Network Topology Definition — 12-cell HetNet layout
# ═══════════════════════════════════════════════════════════════
CELL_TOPOLOGY = [
    {"id": "C01", "label": "Macro-Central",   "cell_type": "Macro", "x": 5.0, "y": 5.0, "radius": 3.0},
    {"id": "C02", "label": "Macro-North",     "cell_type": "Macro", "x": 5.0, "y": 9.0, "radius": 3.0},
    {"id": "C03", "label": "Macro-South",     "cell_type": "Macro", "x": 5.0, "y": 1.0, "radius": 3.0},
    {"id": "C04", "label": "Micro-Stadium",   "cell_type": "Micro", "x": 2.0, "y": 7.0, "radius": 1.5},
    {"id": "C05", "label": "Micro-Hospital",  "cell_type": "Micro", "x": 8.0, "y": 7.0, "radius": 1.5},
    {"id": "C06", "label": "Micro-Factory",   "cell_type": "Micro", "x": 8.0, "y": 3.0, "radius": 1.5},
    {"id": "C07", "label": "Micro-Downtown",  "cell_type": "Micro", "x": 2.0, "y": 3.0, "radius": 1.5},
    {"id": "C08", "label": "Pico-Mall",       "cell_type": "Pico",  "x": 3.5, "y": 5.5, "radius": 0.8},
    {"id": "C09", "label": "Pico-University", "cell_type": "Pico",  "x": 6.5, "y": 5.5, "radius": 0.8},
    {"id": "C10", "label": "Pico-Park",       "cell_type": "Pico",  "x": 3.5, "y": 4.0, "radius": 0.8},
    {"id": "C11", "label": "Femto-Office-A",  "cell_type": "Femto", "x": 6.5, "y": 8.0, "radius": 0.4},
    {"id": "C12", "label": "Femto-Office-B",  "cell_type": "Femto", "x": 1.5, "y": 5.0, "radius": 0.4},
]

# Maps each topology cell to a specific Cell_ID in the dataset.
# Each Cell_ID has ~100 real rows so each cell gets its own distinct KPI profile.
# Layout: Macro IDs 3,4,13 | Micro IDs 2,7,12,14 | Pico IDs 9,11,18 | Femto IDs 1,5
TOPOLOGY_CELL_ID_MAP = {
    "C01": 3,   # Macro-Central   → dataset Cell_ID 3  (Macro)
    "C02": 4,   # Macro-North     → dataset Cell_ID 4  (Macro)
    "C03": 13,  # Macro-South     → dataset Cell_ID 13 (Macro)
    "C04": 2,   # Micro-Stadium   → dataset Cell_ID 2  (Micro)
    "C05": 7,   # Micro-Hospital  → dataset Cell_ID 7  (Micro)
    "C06": 12,  # Micro-Factory   → dataset Cell_ID 12 (Micro)
    "C07": 14,  # Micro-Downtown  → dataset Cell_ID 14 (Micro)
    "C08": 9,   # Pico-Mall       → dataset Cell_ID 9  (Pico)
    "C09": 11,  # Pico-University → dataset Cell_ID 11 (Pico)
    "C10": 18,  # Pico-Park       → dataset Cell_ID 18 (Pico)
    "C11": 1,   # Femto-Office-A  → dataset Cell_ID 1  (Femto)
    "C12": 5,   # Femto-Office-B  → dataset Cell_ID 5  (Femto)
}


def initialize_session_state():
    """Initialize session state variables"""
    if 'agent_states' not in st.session_state:
        st.session_state.agent_states = {
            'intent': {'status': 'waiting', 'output': None},
            'reasoner': {'status': 'waiting', 'output': None},
            'validator': {'status': 'waiting', 'output': None},
            'planner': {'status': 'waiting', 'output': None},
            'monitor': {'status': 'waiting', 'output': None},
            'optimizer': {'status': 'waiting', 'output': None}
        }
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'optimization_log' not in st.session_state:
        st.session_state.optimization_log = []
    if 'current_config' not in st.session_state:
        st.session_state.current_config = None
    if 'conflict_results' not in st.session_state:
        st.session_state.conflict_results = None
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    if 'auto_heal_enabled' not in st.session_state:
        st.session_state.auto_heal_enabled = True
    if 'last_healing_result' not in st.session_state:
        st.session_state.last_healing_result = None
    if 'healing_count' not in st.session_state:
        st.session_state.healing_count = 0
    if 'monitor_cycle_count' not in st.session_state:
        st.session_state.monitor_cycle_count = 0
    # Level 5 Autonomy - Rollback & Audit
    if 'rollback_history' not in st.session_state:
        st.session_state.rollback_history = []
    if 'autonomous_audit_log' not in st.session_state:
        st.session_state.autonomous_audit_log = []
    if 'last_action_state' not in st.session_state:
        st.session_state.last_action_state = None  # For rollback capability
    if 'action_cooldowns' not in st.session_state:
        st.session_state.action_cooldowns = {}  # Track cooldown timers
    if 'rollback_count' not in st.session_state:
        st.session_state.rollback_count = 0
    if 'topology_cell_metrics' not in st.session_state:
        st.session_state.topology_cell_metrics = []
    # Reasoning Layer state
    if 'reasoning_phase' not in st.session_state:
        st.session_state.reasoning_phase = None
    if 'reasoning_questions' not in st.session_state:
        st.session_state.reasoning_questions = []
    if 'reasoning_answers' not in st.session_state:
        st.session_state.reasoning_answers = {}
    if 'partial_intent_result' not in st.session_state:
        st.session_state.partial_intent_result = None


# ═══════════════════════════════════════════════════════════════
# Network Topology — Visualization Functions
# ═══════════════════════════════════════════════════════════════

def generate_topology_metrics():
    """Generate per-cell live metrics by sampling the real dataset."""
    cell_capacity = {'Macro': 1000, 'Micro': 500, 'Pico': 200, 'Femto': 50}
    metrics = []

    for cell_def in CELL_TOPOLOGY:
        ct = cell_def['cell_type']

        # Try to sample from real dataset using this cell's dedicated Cell_ID
        if network_simulator._use_real_data and network_simulator.dataset is not None:
            df = network_simulator.dataset
            dataset_cell_id = TOPOLOGY_CELL_ID_MAP.get(cell_def['id'])
            if dataset_cell_id is not None:
                cell_rows = df[df['Cell_ID'] == dataset_cell_id]
            else:
                cell_rows = df[df['Cell_Type'] == ct]
            type_rows = cell_rows if len(cell_rows) > 0 else df[df['Cell_Type'] == ct]
            if len(type_rows) > 0:
                row = type_rows.sample(1).iloc[0]
                variation = random.uniform(0.90, 1.10)
                throughput = float(row.get('Achieved_Throughput_Mbps', 100)) * variation
                latency = float(row.get('Network_Latency_ms', 20)) * variation
                cell_load = float(row.get('Resource_Utilization', 50.0)) * variation  # Already in %
                cell_load = min(100, cell_load)
                snr = float(row.get('Signal_to_Noise_Ratio_dB', 20)) * variation
                interference = float(row.get('Interference_Level_dB', -100))
                users = int(cell_capacity.get(ct, 500) * (cell_load / 100))
            else:
                throughput = random.uniform(50, 200)
                latency = random.uniform(5, 60)
                cell_load = random.uniform(20, 90)
                snr = random.uniform(10, 35)
                interference = random.uniform(-120, -80)
                users = int(cell_capacity.get(ct, 500) * (cell_load / 100))
        else:
            throughput = random.uniform(50, 200)
            latency = random.uniform(5, 60)
            cell_load = random.uniform(20, 90)
            snr = random.uniform(10, 35)
            interference = random.uniform(-120, -80)
            users = int(cell_capacity.get(ct, 500) * (cell_load / 100))

        # Apply anomaly effect if active
        if network_simulator._anomaly_active:
            latency *= 1.5
            throughput *= 0.7
            cell_load = min(100, cell_load * 1.3)

        # Calculate health
        score = 100
        if latency > 80:
            score -= 30
        elif latency > 50:
            score -= 15
        if throughput < 50:
            score -= 25
        elif throughput < 80:
            score -= 10
        if cell_load > 90:
            score -= 25
        elif cell_load > 80:
            score -= 10

        if score >= 80:
            health = "healthy"
            health_color = "#198754"
        elif score >= 60:
            health = "warning"
            health_color = "#ffc107"
        else:
            health = "critical"
            health_color = "#dc3545"

        metrics.append({
            "id": cell_def['id'],
            "label": cell_def['label'],
            "cell_type": ct,
            "throughput_mbps": round(throughput, 1),
            "latency_ms": round(latency, 1),
            "cell_load_percent": round(cell_load, 1),
            "connected_users": users,
            "snr_db": round(snr, 1),
            "interference_level_db": round(interference, 1),
            "health": health,
            "health_color": health_color,
            "health_score": score,
        })

    st.session_state.topology_cell_metrics = metrics


def create_topology_figure(cell_metrics, highlight_cells=None, compact=False):
    """Build a Plotly figure for the network topology."""
    fig = go.Figure()

    # Use distinctive marker symbols for each cell type
    symbol_map = {'Macro': 'diamond', 'Micro': 'square', 'Pico': 'circle', 'Femto': 'triangle-up'}
    size_map = {'Macro': 50, 'Micro': 40, 'Pico': 32, 'Femto': 26}
    icon_map = {'Macro': '🗼', 'Micro': '📡', 'Pico': '📶', 'Femto': '🏠'}

    # Layer 1: Coverage circles
    for cell_def in CELL_TOPOLOGY:
        r = cell_def['radius']
        fig.add_shape(
            type="circle",
            x0=cell_def['x'] - r, y0=cell_def['y'] - r,
            x1=cell_def['x'] + r, y1=cell_def['y'] + r,
            fillcolor="rgba(13, 110, 253, 0.07)",
            line=dict(color="rgba(13, 110, 253, 0.2)", width=1.5, dash='dot'),
        )

    # Layer 2: Backhaul lines (non-Macro → nearest Macro)
    macro_cells = [c for c in CELL_TOPOLOGY if c['cell_type'] == 'Macro']
    for cell_def in CELL_TOPOLOGY:
        if cell_def['cell_type'] != 'Macro':
            nearest = min(macro_cells, key=lambda m: (m['x'] - cell_def['x'])**2 + (m['y'] - cell_def['y'])**2)
            fig.add_trace(go.Scatter(
                x=[cell_def['x'], nearest['x']], y=[cell_def['y'], nearest['y']],
                mode='lines',
                line=dict(color='#adb5bd', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ))

    # Layer 3: Cell nodes grouped by type
    for cell_type in ['Macro', 'Micro', 'Pico', 'Femto']:
        cells_of_type = [c for c in CELL_TOPOLOGY if c['cell_type'] == cell_type]
        if not cells_of_type:
            continue

        xs = [c['x'] for c in cells_of_type]
        ys = [c['y'] for c in cells_of_type]

        metrics_for_cells = []
        for c in cells_of_type:
            m = next((cm for cm in cell_metrics if cm['id'] == c['id']), None)
            metrics_for_cells.append(m)

        colors = [m['health_color'] if m else '#6c757d' for m in metrics_for_cells]

        hover_texts = []
        text_labels = []
        for c, m in zip(cells_of_type, metrics_for_cells):
            if m:
                hover_texts.append(
                    f"<b>{c['label']}</b> ({c['cell_type']})<br>"
                    f"<b>Health:</b> {m['health'].upper()} ({m['health_score']}/100)<br>"
                    f"─────────────────<br>"
                    f"<b>Throughput:</b> {m['throughput_mbps']:.1f} Mbps<br>"
                    f"<b>Latency:</b> {m['latency_ms']:.1f} ms<br>"
                    f"<b>Cell Load:</b> {m['cell_load_percent']:.1f}%<br>"
                    f"<b>Users:</b> {m['connected_users']}<br>"
                    f"<b>SNR:</b> {m['snr_db']:.1f} dB<br>"
                    f"<b>Interference:</b> {m['interference_level_db']:.1f} dB"
                )
                text_labels.append(f"{m['cell_load_percent']:.0f}%")
            else:
                hover_texts.append(f"<b>{c['label']}</b> (no data)")
                text_labels.append("")

        # Add icons as text (one trace per cell for individual colors)
        icon = icon_map[cell_type]

        for i, (c, m) in enumerate(zip(cells_of_type, metrics_for_cells)):
            cell_color = m['health_color'] if m else '#6c757d'
            load_label = f"{m['cell_load_percent']:.0f}%" if m else ""
            # Short name: remove prefix like "Macro-", "Micro-", etc.
            short_name = c['label'].split('-', 1)[1] if '-' in c['label'] else c['label']
            hover = hover_texts[i]

            # Cell name label above icon
            fig.add_trace(go.Scatter(
                x=[c['x']], y=[c['y'] + 0.8],
                mode='text',
                text=[short_name],
                textfont=dict(size=13, color='#333', family='Arial'),
                showlegend=False,
                hoverinfo='skip',
            ))

            # Icon
            fig.add_trace(go.Scatter(
                x=[c['x']], y=[c['y']],
                mode='text',
                text=[icon],
                textfont=dict(size=size_map[cell_type], color=cell_color),
                name=f"{icon} {cell_type}" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup=cell_type,
                hovertext=[hover],
                hoverinfo='text',
            ))

            # Load % label below icon
            fig.add_trace(go.Scatter(
                x=[c['x']], y=[c['y'] - 0.7],
                mode='text',
                text=[load_label],
                textfont=dict(size=14, color=cell_color, family='Arial Black'),
                showlegend=False,
                hoverinfo='skip',
            ))

    # Layer 4: Highlight rings for affected cells
    if highlight_cells:
        highlight_defs = [c for c in CELL_TOPOLOGY if c['id'] in highlight_cells]
        if highlight_defs:
            fig.add_trace(go.Scatter(
                x=[c['x'] for c in highlight_defs],
                y=[c['y'] for c in highlight_defs],
                mode='markers',
                marker=dict(size=38, color='rgba(255, 193, 7, 0.35)',
                            line=dict(color='#ffc107', width=3)),
                name='Affected',
                hoverinfo='skip',
            ))

    height = 400 if compact else 750
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 10.5]),
        yaxis=dict(visible=False, range=[-1.0, 10.5], scaleanchor="x"),
        height=height,
        showlegend=not compact,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )

    return fig


def render_network_topology():
    """Render the live network topology with auto-refresh option."""
    col_title, col_toggle = st.columns([3, 1])
    with col_title:
        st.subheader("🗼 Live Network Topology")
    with col_toggle:
        auto_refresh = st.checkbox("🔄 Live", value=st.session_state.get('topology_auto_refresh', False), key="topology_auto_refresh")

    generate_topology_metrics()
    cell_metrics = st.session_state.topology_cell_metrics

    fig = create_topology_figure(cell_metrics)
    st.plotly_chart(fig, use_container_width=True, key="topology_main")

    # Summary row
    healthy = sum(1 for m in cell_metrics if m['health'] == 'healthy')
    warning = sum(1 for m in cell_metrics if m['health'] == 'warning')
    critical = sum(1 for m in cell_metrics if m['health'] == 'critical')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cells", len(cell_metrics))
    with col2:
        st.metric("Healthy", healthy)
    with col3:
        st.metric("Warning", warning)
    with col4:
        st.metric("Critical", critical)


# ═══════════════════════════════════════════════════════════════
# Reasoning Layer — Clarification, Feasibility, Risk, Impact
# ═══════════════════════════════════════════════════════════════

def _generate_reasoning_questions(intent_result):
    """Generate clarifying questions — powered by real Groq LLM with keyword fallback."""
    return _llm_questions(intent_result)


def _check_feasibility(intent_result, answers, metrics):
    """Check if the intent is feasible given current network state."""
    intent_type = intent_result.get('intent_type', '')
    score = 100
    constraints = []

    current_load = metrics.get('cell_load_percent', 50)
    current_latency = metrics.get('latency_ms', 30)
    current_throughput = metrics.get('throughput_mbps', 100)

    # Capacity check based on audience/device count
    size_answer = answers.get('audience_size', answers.get('device_count', answers.get('user_count', '')))
    if '50,000' in str(size_answer):
        if current_load > 70:
            score -= 25
            constraints.append("High cell load (>70%) may limit capacity for 50,000+ users")
        else:
            score -= 10
    elif '30,000' in str(size_answer) or '10,000' in str(size_answer) or '1,000+' in str(size_answer):
        if current_load > 80:
            score -= 15
            constraints.append("Cell load above 80% — additional cells may be needed")

    # Bandwidth check
    if current_throughput < 60:
        score -= 15
        constraints.append(f"Current throughput ({current_throughput:.0f} Mbps) is below optimal — bandwidth scaling recommended")

    # Latency check for URLLC needs
    urllc_needed = answers.get('urllc_needed', answers.get('latency_tolerance', ''))
    if 'Yes' in str(urllc_needed) or 'Critical' in str(urllc_needed) or 'Ultra-low' in str(urllc_needed):
        if current_latency > 50:
            score -= 20
            constraints.append(f"Current latency ({current_latency:.0f}ms) too high for URLLC — priority scheduling needed")
        elif current_latency > 20:
            score -= 5

    # Cell availability check
    affected_cells = _map_intent_to_affected_cells(intent_type)
    if st.session_state.topology_cell_metrics:
        affected_metrics = [m for m in st.session_state.topology_cell_metrics if m['id'] in affected_cells]
        overloaded = [m for m in affected_metrics if m['cell_load_percent'] > 85]
        if overloaded:
            score -= 10 * len(overloaded)
            constraints.append(f"{len(overloaded)} affected cell(s) already above 85% load")

    score = max(0, min(100, score))
    return {
        'feasible': score >= 40,
        'feasibility_score': score,
        'constraints': constraints,
    }


def _identify_risks(intent_result, answers, metrics):
    """Identify risks based on intent, answers, and current network state."""
    risks = []
    intent_type = intent_result.get('intent_type', '')
    current_load = metrics.get('cell_load_percent', 50)

    # Overload risk
    size_answer = str(answers.get('audience_size', answers.get('device_count', answers.get('user_count', ''))))
    if '50,000' in size_answer or '10,000' in size_answer or '1,000+' in size_answer:
        severity = 'HIGH' if current_load > 70 else 'MEDIUM'
        risks.append({
            'risk': 'Cell Overload',
            'severity': severity,
            'description': f'High user demand on cells with {current_load:.0f}% current load',
            'mitigation': 'Activate additional cells and enable load balancing',
        })

    # Interference risk
    if intent_type in ('stadium_event', 'concert', 'iot_deployment'):
        risks.append({
            'risk': 'Inter-cell Interference',
            'severity': 'LOW',
            'description': 'Activating new cells may cause interference with neighboring cells',
            'mitigation': 'Enable ICIC (Inter-Cell Interference Coordination)',
        })

    # SLA risk for neighboring services
    if intent_type in ('emergency', 'healthcare'):
        risks.append({
            'risk': 'Neighboring SLA Impact',
            'severity': 'MEDIUM',
            'description': 'Priority reallocation may degrade non-critical services',
            'mitigation': 'Apply minimum QoS guarantees for affected slices',
        })

    # Energy risk
    if intent_type in ('stadium_event', 'concert', 'iot_deployment', 'smart_factory'):
        risks.append({
            'risk': 'Energy Consumption',
            'severity': 'LOW',
            'description': 'Additional cell activation increases power consumption',
            'mitigation': 'Enable energy-saving mode for idle cells after event',
        })

    # Latency risk for real-time use cases
    urllc_needed = str(answers.get('urllc_needed', answers.get('latency_tolerance', '')))
    if 'Yes' in urllc_needed or 'Critical' in urllc_needed:
        current_latency = metrics.get('latency_ms', 30)
        if current_latency > 40:
            risks.append({
                'risk': 'Latency Target Miss',
                'severity': 'HIGH',
                'description': f'Current latency ({current_latency:.0f}ms) may not meet URLLC requirements',
                'mitigation': 'Enable priority scheduling and allocate dedicated PRBs',
            })

    return risks


def _simulate_impact(intent_result, answers, metrics):
    """Simulate the expected impact of the intent on network KPIs."""
    intent_type = intent_result.get('intent_type', '')
    before = {
        'latency_ms': metrics.get('latency_ms', 30),
        'throughput_mbps': metrics.get('throughput_mbps', 100),
        'cell_load_percent': metrics.get('cell_load_percent', 50),
        'connected_users': metrics.get('connected_users', 500),
    }

    # Predict after-state based on intent type and answers
    lat_factor = 1.0
    tp_factor = 1.0
    load_factor = 1.0
    user_add = 0

    if intent_type in ('stadium_event', 'concert'):
        size = str(answers.get('audience_size', '30,000'))
        if '50,000' in size:
            user_add = 50000
            load_factor = 1.35
        elif '30,000' in size:
            user_add = 30000
            load_factor = 1.25
        else:
            user_add = 10000
            load_factor = 1.15
        tp_factor = 1.5
        lat_factor = 0.8 if answers.get('vip_priority') == 'Yes' else 0.9

    elif intent_type in ('emergency', 'healthcare'):
        lat_factor = 0.4  # Aggressive latency reduction
        tp_factor = 1.2
        load_factor = 1.1
        user_add = 500

    elif intent_type in ('iot_deployment', 'smart_factory'):
        device_str = str(answers.get('device_count', '10,000+'))
        if '10,000' in device_str:
            user_add = 10000
            load_factor = 1.3
        elif '1,000' in device_str:
            user_add = 1000
            load_factor = 1.15
        else:
            user_add = 100
            load_factor = 1.05
        tp_factor = 1.2
        latency_tol = str(answers.get('latency_tolerance', ''))
        lat_factor = 0.5 if 'Critical' in latency_tol else 0.8

    elif intent_type in ('gaming', 'video_conferencing'):
        user_str = str(answers.get('user_count', '500'))
        if '1,000' in user_str:
            user_add = 1000
        elif '500' in user_str:
            user_add = 500
        else:
            user_add = 100
        quality = str(answers.get('quality_priority', ''))
        if 'latency' in quality.lower():
            lat_factor = 0.6
            tp_factor = 1.3
        elif 'bandwidth' in quality.lower():
            tp_factor = 1.8
            lat_factor = 0.85
        else:
            lat_factor = 0.75
            tp_factor = 1.5
        load_factor = 1.15

    elif intent_type == 'transportation':
        lat_factor = 0.5
        tp_factor = 1.3
        load_factor = 1.2
        user_add = 2000

    else:
        lat_factor = 0.85
        tp_factor = 1.2
        load_factor = 1.05
        user_add = 200

    predicted_after = {
        'latency_ms': round(before['latency_ms'] * lat_factor, 1),
        'throughput_mbps': round(before['throughput_mbps'] * tp_factor, 1),
        'cell_load_percent': round(min(100, before['cell_load_percent'] * load_factor), 1),
        'connected_users': before['connected_users'] + user_add,
    }

    return {'before': before, 'predicted_after': predicted_after}


def render_reasoning_form():
    """Render the reasoning form with clarifying questions and analysis results."""
    if st.session_state.reasoning_phase != 'questions_ready':
        return

    questions = st.session_state.reasoning_questions
    intent_result = st.session_state.partial_intent_result
    intent_type = intent_result.get('intent_type', 'unknown')

    st.markdown(f'<div style="background:#e8f4fd;border:2px solid #0d6efd;border-radius:10px;padding:1rem;margin:1rem 0;"><b>🤔 Reasoning Agent</b> — Analyzing intent: <b>{intent_type}</b>. Please confirm or adjust the parameters below.</div>', unsafe_allow_html=True)

    # Show questions as a form
    with st.form("reasoning_form"):
        st.subheader("Clarifying Questions")
        answers = {}
        cols = st.columns(2)
        for idx, q in enumerate(questions):
            with cols[idx % 2]:
                default_idx = q['options'].index(q['default']) if q['default'] in q['options'] else 0
                answers[q['id']] = st.radio(
                    f"{q['icon']} {q['question']}",
                    options=q['options'],
                    index=default_idx,
                    key=f"reasoning_q_{q['id']}",
                )

        submitted = st.form_submit_button("🚀 Confirm & Execute", type="primary", use_container_width=True)

    if submitted:
        st.session_state.reasoning_answers = answers
        st.session_state.reasoning_phase = 'confirmed'
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Configuration Plan — Visual Display Functions
# ═══════════════════════════════════════════════════════════════

def _map_intent_to_affected_cells(intent_type, entities=None):
    """Map intent type to affected topology cell IDs using the LLM.
    Falls back to keyword matching if LLM is unavailable."""
    return map_intent_to_cells_with_llm(intent_type, entities or {})


def _estimate_expected_impact(config, intent_output):
    """Generate human-readable predicted impact descriptions."""
    slice_info = config.get('network_slice', {})
    ran_info = config.get('ran_configuration', {})
    capacity = config.get('expected_capacity', {})
    slice_type = slice_info.get('type', 'eMBB')
    impacts = []

    # Slice type specific impacts
    if slice_type == 'URLLC':
        impacts.append({"text": "URLLC slice activated — Ultra-Reliable Low-Latency mode", "icon": "🚨"})
    elif slice_type == 'mMTC':
        impacts.append({"text": "mMTC slice activated — Massive IoT device support", "icon": "🔌"})

    bw = slice_info.get('allocated_bandwidth_mbps', 100)
    if bw > 100:
        impacts.append({"text": f"Bandwidth increased to {bw:.0f} Mbps (+{((bw / 100) - 1) * 100:.0f}%)", "icon": "📶"})
    elif bw < 100:
        impacts.append({"text": f"Bandwidth optimized to {bw:.0f} Mbps (efficiency mode)", "icon": "📶"})

    lat = slice_info.get('latency_target_ms', 50)
    if lat < 1:
        impacts.append({"text": f"Ultra-low latency: {lat:.1f} ms (99% reduction)", "icon": "⚡"})
    elif lat < 50:
        impacts.append({"text": f"Latency target reduced to {lat:.0f} ms ({((50 - lat) / 50) * 100:.0f}% improvement)", "icon": "⚡"})

    cells = ran_info.get('active_cells', 20)
    if cells > 20:
        impacts.append({"text": f"{cells - 20} additional cells activated (total: {cells})", "icon": "🗼"})

    users = capacity.get('max_users', 10000)
    per_user = capacity.get('expected_throughput_per_user_mbps', 1)
    impacts.append({"text": f"Capacity for {users:,} users at {per_user:.1f} Mbps/user", "icon": "👥"})

    mimo = ran_info.get('mimo_configuration', '2x2')
    if mimo != '2x2':
        impacts.append({"text": f"MIMO upgraded to {mimo} for improved spectral efficiency", "icon": "📡"})

    scheduler = ran_info.get('scheduler_type', 'proportional_fair')
    if scheduler == 'strict_priority':
        impacts.append({"text": "Strict priority scheduling for critical traffic", "icon": "🎯"})
    elif scheduler == 'weighted_fair':
        impacts.append({"text": "Weighted fair scheduling for balanced performance", "icon": "⚖️"})

    return impacts


def _build_configuration_plan():
    """Build a structured configuration plan from agent outputs."""
    config = st.session_state.current_config
    intent_output = st.session_state.agent_states['intent']['output']
    validator_output = st.session_state.agent_states['validator']['output']

    if not config:
        return None

    slice_info = config.get('network_slice', {})
    ran_info = config.get('ran_configuration', {})
    qos_info = config.get('qos_parameters', {})
    entities = intent_output.get('entities', {}) if intent_output else {}

    # Parameter changes - show meaningful current vs planned values
    slice_type = slice_info.get('type', 'eMBB')
    latency_val = slice_info.get('latency_target_ms', 50)
    priority_val = slice_info.get('priority', 5)
    qos_5qi = qos_info.get('5qi', 9)
    scheduler = ran_info.get('scheduler_type', 'proportional_fair')

    parameter_changes = [
        {"parameter": "Network Slice", "from": "eMBB (Standard)", "to": f"{slice_type} — {slice_info.get('name', '')[:30]}"},
        {"parameter": "Bandwidth", "from": "100 Mbps", "to": f"{slice_info.get('allocated_bandwidth_mbps', 100):.0f} Mbps"},
        {"parameter": "Latency Target", "from": "50 ms", "to": f"{latency_val:.1f} ms" if latency_val < 1 else f"{latency_val:.0f} ms"},
        {"parameter": "Priority Level", "from": "5 (Normal)", "to": f"{priority_val} ({'Highest' if priority_val == 1 else 'High' if priority_val <= 3 else 'Normal'})"},
        {"parameter": "Active Cells", "from": "20", "to": str(ran_info.get('active_cells', 20))},
        {"parameter": "MIMO Config", "from": "2x2", "to": ran_info.get('mimo_configuration', '2x2')},
        {"parameter": "Scheduler", "from": "proportional_fair", "to": scheduler},
        {"parameter": "5QI (QoS Class)", "from": "9 (Best Effort)", "to": f"{qos_5qi} ({'Mission Critical' if qos_5qi <= 2 else 'Real-Time' if qos_5qi <= 4 else 'Standard'})"},
    ]

    # Affected cells — LLM selects the most relevant topology cells
    intent_type = intent_output.get('intent_type', '') if intent_output else ''
    affected_cells = _map_intent_to_affected_cells(intent_type, entities)

    # Timeline
    time_entity = entities.get('time', '')
    if not time_entity or time_entity in ('immediate', 'now'):
        timeline = "Immediately"
        timeline_detail = "Configuration applied upon execution"
    elif time_entity == 'tomorrow':
        timeline = "Scheduled: Tomorrow"
        timeline_detail = "Configuration queued for deployment tomorrow"
    else:
        timeline = f"Scheduled: {time_entity}"
        timeline_detail = f"Configuration queued for {time_entity}"

    # Confidence & risk
    confidence = 0.0
    risk_level = "safe"
    warnings = []
    if validator_output:
        conf_details = validator_output.get('confidence_details', {})
        confidence = conf_details.get('actual', validator_output.get('confidence', 0.0))
        risk_level = validator_output.get('risk_level', 'safe')
        warnings = validator_output.get('warnings', [])

    # Expected impact
    expected_impact = _estimate_expected_impact(config, intent_output)

    return {
        "parameter_changes": parameter_changes,
        "affected_cells": affected_cells,
        "expected_impact": expected_impact,
        "timeline": timeline,
        "timeline_detail": timeline_detail,
        "confidence": confidence,
        "risk_level": risk_level,
        "warnings": warnings,
        "slice_type": slice_info.get('type', 'eMBB'),
        "priority": slice_info.get('priority', 5),
        "intent_type": intent_type,
    }


def render_configuration_plan():
    """Render the structured configuration plan after intent processing."""
    if not st.session_state.current_config:
        return

    plan = _build_configuration_plan()
    if not plan:
        return

    st.header("📋 Configuration Plan")

    # ── Top banner: Confidence | Risk | Timeline | Slice ──
    conf = plan['confidence']
    risk = plan['risk_level']
    risk_colors = {
        'safe': ('#198754', '#d4edda'),
        'caution': ('#ffc107', '#fff3cd'),
        'blocked': ('#dc3545', '#f8d7da'),
    }
    risk_color, risk_bg = risk_colors.get(risk, ('#6c757d', '#f8f9fa'))

    conf_color = '#198754' if conf >= 0.9 else '#ffc107' if conf >= 0.7 else '#dc3545'

    st.markdown(f"""
    <div style="display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 140px; background: #f0f7f0; border: 2px solid {conf_color};
                    border-radius: 10px; padding: 0.8rem; text-align: center;">
            <div style="font-size: 1.8rem; font-weight: bold; color: {conf_color};">{conf:.0%}</div>
            <div style="color: #333; font-size: 0.85rem;">Confidence</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: {risk_bg}; border: 2px solid {risk_color};
                    border-radius: 10px; padding: 0.8rem; text-align: center;">
            <div style="font-size: 1.4rem; font-weight: bold; color: {risk_color};">{risk.upper()}</div>
            <div style="color: #333; font-size: 0.85rem;">Risk Level</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: #e8f4fd; border: 2px solid #0d6efd;
                    border-radius: 10px; padding: 0.8rem; text-align: center;">
            <div style="font-size: 1.1rem; font-weight: bold; color: #0d6efd;">{plan['timeline']}</div>
            <div style="color: #333; font-size: 0.85rem;">Timeline</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: #f8f9fa; border: 2px solid #6c757d;
                    border-radius: 10px; padding: 0.8rem; text-align: center;">
            <div style="font-size: 1.4rem; font-weight: bold; color: #333;">{plan['slice_type']}</div>
            <div style="color: #333; font-size: 0.85rem;">Slice Type (P{plan['priority']})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: Parameter Changes | Affected Elements ──
    col_plan, col_topo = st.columns([3, 2])

    with col_plan:
        st.subheader("Parameter Changes")
        rows_html = ""
        for change in plan['parameter_changes']:
            from_val = change['from'].split(' (')[0]
            to_val = change['to'].split(' —')[0].split(' (')[0]
            changed = from_val != to_val
            arrow = "&#10132;" if changed else "="
            row_bg = "#fff3cd" if changed else "#f8f9fa"
            to_weight = "bold" if changed else "normal"
            rows_html += f"""<tr style="background: {row_bg};"><td style="padding: 0.4rem 0.6rem; font-weight: 600; color: #333; border-bottom: 1px solid #e9ecef;">{change['parameter']}</td><td style="padding: 0.4rem 0.6rem; color: #666; border-bottom: 1px solid #e9ecef;">{change['from']}</td><td style="padding: 0.4rem; text-align: center; font-size: 1.1rem; color: #0d6efd; border-bottom: 1px solid #e9ecef;">{arrow}</td><td style="padding: 0.4rem 0.6rem; color: #000; font-weight: {to_weight}; border-bottom: 1px solid #e9ecef;">{change['to']}</td></tr>"""

        table_html = f"""<table style="width: 100%; border-collapse: collapse; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden;">
<thead>
<tr style="background: #e9ecef;">
<th style="padding: 0.5rem 0.6rem; text-align: left; color: #333;">Parameter</th>
<th style="padding: 0.5rem 0.6rem; text-align: left; color: #333;">Current</th>
<th style="padding: 0.5rem; text-align: center; color: #333; width: 30px;"></th>
<th style="padding: 0.5rem 0.6rem; text-align: left; color: #333;">Planned</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>"""
        st.markdown(table_html, unsafe_allow_html=True)

    with col_topo:
        st.subheader("Affected Elements")
        if not st.session_state.topology_cell_metrics:
            generate_topology_metrics()
        cell_metrics = st.session_state.topology_cell_metrics
        fig = create_topology_figure(cell_metrics, highlight_cells=plan['affected_cells'], compact=True)
        st.plotly_chart(fig, use_container_width=True, key="topology_plan")

        # List affected cell names
        affected_names = [c['label'] for c in CELL_TOPOLOGY if c['id'] in plan['affected_cells']]
        st.caption(f"Affected: {', '.join(affected_names)}")

    # ── Expected Impact ──
    st.subheader("Expected Impact")
    impact_html = ""
    for impact in plan['expected_impact']:
        impact_html += f"""
        <div style="background: #d4edda; border-left: 4px solid #198754; padding: 0.5rem 1rem;
                    margin-bottom: 0.4rem; border-radius: 0 6px 6px 0; color: #333; font-size: 0.95rem;">
            {impact['icon']} {impact['text']}
        </div>
        """
    st.markdown(impact_html, unsafe_allow_html=True)

    # ── Warnings ──
    if plan['warnings']:
        st.markdown("")
        for w in plan['warnings']:
            st.warning(w)


def render_header():
    """Render the application header"""
    st.title("📡 Autonomous Intent-Based 5G-Advanced Network Optimizer")
    st.markdown("Agentic AI for Real-Time Optimization of 5G-Advanced RAN. Enter your network requirements in natural language and watch the AI agents work together to configure and optimize the network.")


def render_sidebar():
    """Render the sidebar with settings and info"""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Event simulation
        st.subheader("Event Simulation")
        event_type = st.selectbox(
            "Simulate Event",
            ["None", "Stadium Event", "Concert", "Emergency", "IoT Deployment",
             "Healthcare", "Transportation", "Smart Factory", "Video Conferencing", "Gaming"],
            key="event_type"
        )

        if event_type != "None":
            event_map = {
                "Stadium Event": "stadium",
                "Concert": "concert",
                "Emergency": "emergency",
                "IoT Deployment": "iot_deployment",
                "Healthcare": "healthcare",
                "Transportation": "transportation",
                "Smart Factory": "smart_factory",
                "Video Conferencing": "video_conferencing",
                "Gaming": "gaming"
            }
            if st.button("Start Event"):
                network_simulator.start_event(event_map[event_type])
                st.success(f"Started: {event_type}")
        else:
            if st.button("Stop Event"):
                network_simulator.stop_event()
                st.info("Event stopped")

        st.divider()

        # Network info
        st.subheader("Network Status")
        status = network_simulator.get_status_summary()
        health = status['overall_health']
        health_color = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴'
        }.get(health, '⚪')

        st.metric("Health", f"{health_color} {health.upper()}")
        st.metric("Active Slices", status['active_slices'])
        st.metric("Event Active", "Yes" if status['event_active'] else "No")

        st.divider()

        # About section
        st.subheader("About")
        st.markdown("""
        **5G-Advanced Features:**
        - Intent-Based Networking
        - AI-Native Optimization
        - Predictive QoS
        - Self-Organizing Network

        **3GPP Release 18 Compliant**
        """)


def render_intent_input():
    """Render the intent input section"""
    st.header("📝 Enter Your Intent")

    # Initialize demo text holder
    if 'demo_intent' not in st.session_state:
        st.session_state.demo_intent = ""

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**Quick Examples:**")
        if st.button("📺 Stadium Event"):
            st.session_state.demo_intent = "Optimize for live streaming at the stadium tomorrow evening with high quality video support"
            st.rerun()
        if st.button("🚨 Emergency"):
            st.session_state.demo_intent = "Emergency priority for communications in the hospital area immediately"
            st.rerun()
        if st.button("🏭 IoT Sensors"):
            st.session_state.demo_intent = "Deploy connectivity for 10000 IoT sensors in the industrial zone"
            st.rerun()

    with col1:
        user_intent = st.text_area(
            "Describe your network requirements:",
            value=st.session_state.demo_intent,
            placeholder="Example: Optimize for live streaming at the stadium tomorrow from 7-10 PM with excellent quality",
            height=100,
        )

    return user_intent


def _get_agent_summary(agent_key: str) -> str:
    """Extract a 1-line summary from agent output for pipeline display"""
    output = st.session_state.agent_states[agent_key].get('output')
    if not output:
        return ""

    if agent_key == 'intent':
        intent_type = output.get('intent_type', '?')
        confidence = output.get('confidence', 0)
        return f"{intent_type} ({confidence:.0%})"
    elif agent_key == 'reasoner':
        f_score = output.get('feasibility', {}).get('feasibility_score', '?')
        n_risks = len(output.get('risks', []))
        return f"Feasibility: {f_score}/100, {n_risks} risks"
    elif agent_key == 'validator':
        risk = output.get('risk_level', '?')
        if not output.get('approved', True):
            return "BLOCKED"
        return f"Risk: {risk}"
    elif agent_key == 'planner':
        ns = output.get('network_slice', {})
        return f"{ns.get('type', '?')} {ns.get('allocated_bandwidth_mbps', '?')}Mbps"
    elif agent_key == 'monitor':
        health = output.get('overall_status', '?')
        return f"Health: {health}"
    elif agent_key == 'optimizer':
        msg = output.get('message', '')
        if 'No optimization' in msg:
            return "No action needed"
        action = output.get('action', '')
        return f"Applied: {action}" if action else "Optimized"
    return ""


def render_agent_activity():
    """Render the agent pipeline visualization"""
    st.header("🤖 Agent Pipeline")

    agents = [
        {'key': 'intent', 'icon': '🧠', 'name': 'Intent Interpreter', 'role': 'NLP Analysis'},
        {'key': 'reasoner', 'icon': '🤔', 'name': 'Reasoner', 'role': 'Analysis & Verification'},
        {'key': 'validator', 'icon': '🛡️', 'name': 'Intent Validator', 'role': 'Safety Gate'},
        {'key': 'planner', 'icon': '📋', 'name': 'Planner', 'role': 'Config Generator'},
        {'key': 'monitor', 'icon': '📊', 'name': 'Monitor', 'role': 'Network Analysis'},
        {'key': 'optimizer', 'icon': '⚡', 'name': 'Optimizer', 'role': 'Auto-Optimization'},
    ]

    status_styles = {
        'waiting': {'bg': '#f0f0f0', 'border': '#cccccc', 'text': '#888888', 'icon': '⏸️', 'label': 'Waiting', 'arrow': '#cccccc'},
        'running': {'bg': '#cce5ff', 'border': '#0d6efd', 'text': '#0d6efd', 'icon': '🔄', 'label': 'Running', 'arrow': '#0d6efd'},
        'completed': {'bg': '#d4edda', 'border': '#28a745', 'text': '#155724', 'icon': '✅', 'label': 'Done', 'arrow': '#28a745'},
        'error': {'bg': '#f8d7da', 'border': '#dc3545', 'text': '#721c24', 'icon': '❌', 'label': 'Error', 'arrow': '#dc3545'},
    }

    nodes_html = ""
    for i, agent in enumerate(agents):
        status = st.session_state.agent_states[agent['key']]['status']
        style = status_styles.get(status, status_styles['waiting'])
        summary = _get_agent_summary(agent['key'])

        # Arrow between nodes (not before the first one)
        if i > 0:
            prev_status = st.session_state.agent_states[agents[i-1]['key']]['status']
            arrow_color = status_styles.get(prev_status, status_styles['waiting'])['arrow']
            nodes_html += f'<div style="display:flex;align-items:center;padding:0 2px;"><span style="font-size:1.3rem;color:{arrow_color};">&#9654;</span></div>'

        summary_html = f'<div style="font-size:0.7rem;color:{style["text"]};margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px;">{summary}</div>' if summary else ''

        nodes_html += f'''<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">
<div style="background:{style['bg']};border:2px solid {style['border']};border-radius:12px;padding:10px 8px;text-align:center;width:100%;max-width:140px;{'animation:pulse 1.5s ease-in-out infinite;' if status == 'running' else ''}">
<div style="font-size:1.5rem;">{agent['icon']}</div>
<div style="font-size:0.75rem;font-weight:bold;color:#333;margin:4px 0 2px;">{agent['name']}</div>
<div style="font-size:0.65rem;color:#666;margin-bottom:4px;">{agent['role']}</div>
<div style="font-size:0.7rem;font-weight:bold;color:{style['text']};">{style['icon']} {style['label']}</div>{summary_html}
</div>
</div>'''

    pipeline_html = f'''<style>@keyframes pulse {{0%,100%{{box-shadow:0 0 5px rgba(13,110,253,0.3);}}50%{{box-shadow:0 0 15px rgba(13,110,253,0.6);}}}}</style>
<div style="display:flex;align-items:stretch;justify-content:center;gap:0;padding:10px 0;">{nodes_html}</div>'''

    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Expandable output details
    any_completed = any(st.session_state.agent_states[a['key']]['status'] == 'completed' for a in agents)
    if any_completed:
        with st.expander("View Detailed Agent Outputs"):
            for agent in agents:
                output = st.session_state.agent_states[agent['key']].get('output')
                if output and st.session_state.agent_states[agent['key']]['status'] == 'completed':
                    st.subheader(f"{agent['icon']} {agent['name']}")
                    st.json(output)


# ==========================================================================
# KPI MONITORING, ANOMALY ALERTS & SELF-HEALING
# ==========================================================================

AUTO_HEAL_PARAMS = {
    'scale_bandwidth': {'change_mbps': 50},
    'activate_cell': {'count': 2},
    'adjust_priority': {'slice_id': 'default', 'priority': 1},
    'modify_qos': {'latency_target_ms': 30},
    'energy_saving': {'mode': 'moderate'},
}

ACTION_METRIC_MAP = {
    'scale_bandwidth': ['latency_ms', 'throughput_mbps', 'latency_spike'],
    'activate_cell': ['cell_load_percent'],
    'modify_qos': ['latency_ms', 'packet_loss_percent'],
    'adjust_priority': ['latency_ms'],
    'energy_saving': [],
}


def _process_alerts(status_result):
    """Convert violations/anomalies into alert entries. Auto-resolve cleared conditions."""
    now = datetime.now()
    current_signatures = set()

    for v in status_result.get('violations', []):
        sig = f"{v['metric']}_{v['severity']}"
        current_signatures.add(sig)

        already_active = any(
            a.get('signature') == sig and not a['resolved']
            for a in st.session_state.alert_history
        )
        if not already_active:
            st.session_state.alert_history.append({
                'id': f"alert_{int(now.timestamp()*1000)}_{v['metric']}",
                'signature': sig,
                'timestamp': now,
                'severity': v['severity'],
                'type': 'violation',
                'message': v['message'],
                'metric': v['metric'],
                'value': v.get('current_value'),
                'threshold': v.get('threshold'),
                'resolved': False,
                'resolved_at': None,
                'resolution_action': None,
                'auto_healed': False,
            })

    for a in status_result.get('anomalies', []):
        sig = f"anomaly_{a['type']}"
        current_signatures.add(sig)

        already_active = any(
            al.get('signature') == sig and not al['resolved']
            for al in st.session_state.alert_history
        )
        if not already_active:
            st.session_state.alert_history.append({
                'id': f"alert_{int(now.timestamp()*1000)}_{a['type']}",
                'signature': sig,
                'timestamp': now,
                'severity': 'critical' if a.get('severity') == 'high' else 'warning',
                'type': 'anomaly',
                'message': a['description'],
                'metric': a['type'],
                'value': None,
                'threshold': None,
                'resolved': False,
                'resolved_at': None,
                'resolution_action': None,
                'auto_healed': False,
            })

    # Auto-resolve alerts whose condition cleared
    for alert in st.session_state.alert_history:
        if not alert['resolved'] and alert.get('signature') not in current_signatures:
            alert['resolved'] = True
            alert['resolved_at'] = now
            if not alert['resolution_action']:
                alert['resolution_action'] = 'self_resolved'

    st.session_state.active_alerts = [
        a for a in st.session_state.alert_history if not a['resolved']
    ]

    # Cap history
    if len(st.session_state.alert_history) > 100:
        st.session_state.alert_history = st.session_state.alert_history[-100:]


### ---------------------------------------------------------------------------
### Level 5 Autonomous Functions: Audit Logging & Auto-Rollback
### ---------------------------------------------------------------------------

def _log_autonomous_action(action_type: str, action_data: dict, result: dict, source: str = "auto_heal"):
    """
    Log every autonomous action for audit trail (Level 5 requirement).

    This creates a complete record of all AI decisions for:
    - Regulatory compliance
    - Post-incident analysis
    - System behavior verification
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "autonomy_level": AUTONOMY_LEVEL,
        "action_type": action_type,
        "source": source,
        "action_data": action_data,
        "result": {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "execution_details": result.get("execution_details", {}),
        },
        "system_state": {
            "active_alerts": len(st.session_state.active_alerts),
            "healing_count": st.session_state.healing_count,
            "rollback_count": st.session_state.rollback_count,
        },
        "rollback_eligible": True,  # Can be rolled back if needed
    }

    st.session_state.autonomous_audit_log.append(log_entry)

    # Keep last 500 entries
    if len(st.session_state.autonomous_audit_log) > 500:
        st.session_state.autonomous_audit_log = st.session_state.autonomous_audit_log[-500:]

    return log_entry


def _check_action_cooldown(action: str) -> bool:
    """Check if an action is still in cooldown period."""
    # Handle case when running outside Streamlit (e.g., testing)
    try:
        cooldowns = st.session_state.action_cooldowns
    except (AttributeError, KeyError):
        return True  # No cooldown tracking available, allow action

    if action not in cooldowns:
        return True  # No cooldown, can execute

    last_execution = cooldowns[action]
    cooldown_config = ALLOWED_HEALING_ACTIONS.get(action, {})
    cooldown_seconds = cooldown_config.get("cooldown_seconds", 60)

    elapsed = (datetime.now() - last_execution).total_seconds()
    return elapsed >= cooldown_seconds


def _set_action_cooldown(action: str):
    """Set cooldown timer for an action."""
    st.session_state.action_cooldowns[action] = datetime.now()


def _validate_healing_action(action: str, params: dict) -> tuple:
    """
    Validate that a healing action is within allowed bounds.
    Returns (is_valid, reason)
    """
    if action not in ALLOWED_HEALING_ACTIONS:
        return False, f"Action '{action}' not in allowed list"

    action_config = ALLOWED_HEALING_ACTIONS[action]

    if not action_config.get("enabled", True):
        return False, f"Action '{action}' is disabled"

    if not _check_action_cooldown(action):
        return False, f"Action '{action}' is in cooldown period"

    # Specific validations per action type
    if action == "scale_bandwidth":
        change = abs(params.get("change_mbps", 0))
        max_change = action_config.get("max_change", 100)
        if change > max_change:
            return False, f"Bandwidth change {change} exceeds max {max_change}"

    elif action == "activate_cell":
        count = params.get("count", 1)
        max_count = action_config.get("max_count", 5)
        if count > max_count:
            return False, f"Cell count {count} exceeds max {max_count}"

    return True, "Action validated"


def _save_state_for_rollback(before_metrics: dict, actions: list):
    """Save current state to enable rollback if needed."""
    st.session_state.last_action_state = {
        "timestamp": datetime.now(),
        "before_metrics": before_metrics.copy(),
        "actions_taken": actions.copy(),
        "rollback_attempted": False,
    }


def _check_and_rollback(after_metrics: dict) -> dict:
    """
    Check if healing made things worse and rollback if needed.
    This is the core of Level 5 auto-correction.

    Returns rollback result dict.
    """
    if not ROLLBACK_CONFIG["enabled"]:
        return {"needed": False, "reason": "Rollback disabled"}

    if not st.session_state.last_action_state:
        return {"needed": False, "reason": "No previous state to compare"}

    state = st.session_state.last_action_state

    if state.get("rollback_attempted"):
        return {"needed": False, "reason": "Already attempted rollback for this action"}

    before = state["before_metrics"]

    # Calculate health scores
    before_health = _calculate_health_score(before)
    after_health = _calculate_health_score(after_metrics)
    health_drop = before_health - after_health

    # Check if rollback needed
    needs_rollback = False
    rollback_reasons = []

    # Criterion 1: Health score dropped significantly
    if health_drop > ROLLBACK_CONFIG["health_drop_threshold"]:
        needs_rollback = True
        rollback_reasons.append(f"Health dropped by {health_drop} points ({before_health} → {after_health})")

    # Criterion 2: Any critical metric worsened significantly
    degradation_threshold = ROLLBACK_CONFIG["metric_degradation_percent"] / 100

    # Check latency (higher is worse)
    if before.get("latency_ms", 0) > 0:
        latency_change = (after_metrics.get("latency_ms", 0) - before["latency_ms"]) / before["latency_ms"]
        if latency_change > degradation_threshold:
            needs_rollback = True
            rollback_reasons.append(f"Latency worsened by {latency_change:.0%}")

    # Check throughput (lower is worse)
    if before.get("throughput_mbps", 0) > 0:
        throughput_change = (before["throughput_mbps"] - after_metrics.get("throughput_mbps", 0)) / before["throughput_mbps"]
        if throughput_change > degradation_threshold:
            needs_rollback = True
            rollback_reasons.append(f"Throughput dropped by {throughput_change:.0%}")

    if not needs_rollback:
        return {
            "needed": False,
            "reason": "Metrics improved or stable",
            "health_before": before_health,
            "health_after": after_health,
        }

    # Execute rollback
    st.session_state.last_action_state["rollback_attempted"] = True
    rollback_actions = []

    for action_record in state["actions_taken"]:
        action = action_record["action"]
        reverse_action = _get_reverse_action(action, action_record.get("result", {}))

        if reverse_action:
            try:
                reverse_result = execute_action(reverse_action["action"], reverse_action["params"])
                rollback_actions.append({
                    "original_action": action,
                    "reverse_action": reverse_action["action"],
                    "success": reverse_result.get("success", False),
                })
            except Exception as e:
                rollback_actions.append({
                    "original_action": action,
                    "reverse_action": reverse_action["action"],
                    "success": False,
                    "error": str(e),
                })

    st.session_state.rollback_count += 1

    # Log the rollback
    rollback_entry = {
        "timestamp": datetime.now(),
        "reasons": rollback_reasons,
        "health_before": before_health,
        "health_after": after_health,
        "actions_rolled_back": rollback_actions,
        "success": all(a.get("success", False) for a in rollback_actions),
    }
    st.session_state.rollback_history.append(rollback_entry)

    # Audit log
    _log_autonomous_action(
        "rollback",
        {"reasons": rollback_reasons, "actions": rollback_actions},
        {"success": rollback_entry["success"], "message": "Auto-rollback executed"},
        source="auto_rollback"
    )

    return {
        "needed": True,
        "executed": True,
        "reasons": rollback_reasons,
        "actions_rolled_back": rollback_actions,
        "health_before": before_health,
        "health_after": after_health,
    }


def _get_reverse_action(action: str, original_result: dict) -> dict:
    """Get the reverse action to undo a healing action."""
    reverse_map = {
        "scale_bandwidth": {
            "action": "scale_bandwidth",
            "params": {"change_mbps": -original_result.get("parameters", {}).get("change_mbps", 50)}
        },
        "activate_cell": {
            "action": "energy_saving",  # Deactivate cells
            "params": {"mode": "aggressive"}
        },
        "modify_qos": {
            "action": "modify_qos",
            "params": {"reset": True}
        },
    }
    return reverse_map.get(action)


def _calculate_health_score(metrics: dict) -> int:
    """Calculate health score from metrics (simplified version)."""
    score = 100

    # Latency penalty
    latency = metrics.get("latency_ms", 50)
    if latency > 100:
        score -= 30
    elif latency > 80:
        score -= 15
    elif latency > 50:
        score -= 5

    # Throughput penalty
    throughput = metrics.get("throughput_mbps", 100)
    if throughput < 30:
        score -= 30
    elif throughput < 60:
        score -= 15
    elif throughput < 100:
        score -= 5

    # Packet loss penalty
    packet_loss = metrics.get("packet_loss_percent", 0)
    if packet_loss > 1.0:
        score -= 25
    elif packet_loss > 0.1:
        score -= 10

    # Cell load penalty
    cell_load = metrics.get("cell_load_percent", 50)
    if cell_load > 95:
        score -= 20
    elif cell_load > 85:
        score -= 10

    return max(0, min(100, score))



def _execute_auto_healing(status_result, before_metrics):
    """
    Execute autonomous corrective actions for detected issues.

    Level 5 Enhancements:
    - Validates each action against ALLOWED_HEALING_ACTIONS bounds
    - Logs all actions to audit trail
    - Saves state for potential rollback
    - Checks results and auto-rollbacks if metrics worsen
    """
    recommendations = status_result.get('recommendations', [])
    if not recommendations:
        return False

    before_health = status_result.get('health_score', 0)
    actions_taken = []
    actions_skipped = []

    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_recs = sorted(recommendations, key=lambda r: priority_order.get(r.get('priority', 'low'), 9))

    # Save state BEFORE taking any action (for rollback)
    _save_state_for_rollback(before_metrics, [])

    for rec in sorted_recs[:2]:
        action = rec['action']
        params = AUTO_HEAL_PARAMS.get(action, {})

        # ═══ Level 5: Validate action against bounds ═══
        is_valid, validation_reason = _validate_healing_action(action, params)
        if not is_valid:
            actions_skipped.append({
                'action': action,
                'reason': validation_reason,
                'recommendation': rec,
            })
            # Log skipped action
            _log_autonomous_action(
                action,
                {"params": params, "recommendation": rec},
                {"success": False, "message": f"Skipped: {validation_reason}"},
                source="auto_heal_skipped"
            )
            continue

        try:
            result = execute_action(action, params)

            action_record = {
                'action': action,
                'result': result,
                'reason': rec.get('reason', ''),
                'expected_improvement': rec.get('expected_improvement', ''),
                'params': params,
            }
            actions_taken.append(action_record)

            # ═══ Level 5: Log to audit trail ═══
            _log_autonomous_action(action, {"params": params}, result, source="auto_heal")

            # ═══ Level 5: Set cooldown for this action ═══
            _set_action_cooldown(action)

            addressed = ACTION_METRIC_MAP.get(action, [])
            for alert in st.session_state.active_alerts:
                if alert.get('metric', '') in addressed:
                    alert['resolved'] = True
                    alert['resolved_at'] = datetime.now()
                    alert['resolution_action'] = action
                    alert['auto_healed'] = True

            st.session_state.healing_count += 1

        except Exception as e:
            action_record = {
                'action': action,
                'result': {'success': False, 'error': str(e)},
                'reason': rec.get('reason', ''),
                'expected_improvement': '',
            }
            actions_taken.append(action_record)

            # Log failed action
            _log_autonomous_action(
                action,
                {"params": params},
                {"success": False, "message": str(e)},
                source="auto_heal_error"
            )

    # Update saved state with actual actions taken
    _save_state_for_rollback(before_metrics, actions_taken)

    # Capture after-state
    after_metrics_result = get_metrics()
    after_status = check_status(after_metrics_result)
    after_metrics = after_metrics_result['metrics']

    # ═══ Level 5: Check if rollback needed ═══
    rollback_result = _check_and_rollback(after_metrics)

    st.session_state.last_healing_result = {
        'timestamp': datetime.now(),
        'actions_taken': actions_taken,
        'actions_skipped': actions_skipped,
        'before_metrics': before_metrics,
        'after_metrics': after_metrics,
        'before_health': before_health,
        'after_health': after_status.get('health_score', 0),
        'rollback': rollback_result,  # Include rollback info
        'autonomy_level': AUTONOMY_LEVEL,
    }

    for at in actions_taken:
        st.session_state.optimization_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': at['action'],
            'result': at['result'],
            'source': 'auto_heal',
            'autonomy_level': AUTONOMY_LEVEL,
            'rolled_back': rollback_result.get('needed', False),
        })

    st.session_state.active_alerts = [
        a for a in st.session_state.alert_history if not a['resolved']
    ]

    return True


def _render_monitor_header(status_result):
    """Render health badge and summary stats."""
    health = status_result.get('health_score', 0)
    overall = status_result.get('overall_status', 'healthy')

    color_map = {'healthy': '#198754', 'warning': '#ffc107', 'critical': '#dc3545'}
    label_map = {'healthy': 'HEALTHY', 'warning': 'WARNING', 'critical': 'CRITICAL'}
    color = color_map.get(overall, '#6c757d')
    label = label_map.get(overall, 'UNKNOWN')

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="background: {color}; color: {'white' if overall != 'warning' else '#000'};
                    padding: 0.5rem 1.5rem; border-radius: 25px; font-size: 1.3rem;
                    font-weight: bold; display: inline-block;">
            {health}/100 {label}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Monitor Cycles", st.session_state.monitor_cycle_count)
    with col3:
        st.metric("Active Alerts", len(st.session_state.active_alerts))
    with col4:
        st.metric("Auto-Heals", st.session_state.healing_count)


def _render_monitor_controls():
    """Render auto-refresh and self-healing toggles."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.checkbox("Auto-refresh (5s)", value=False, key="monitor_auto_refresh")

    with col2:
        st.session_state.auto_heal_enabled = st.checkbox(
            "Autonomous Self-Healing",
            value=st.session_state.auto_heal_enabled,
            key="auto_heal_toggle",
        )

    with col3:
        if st.session_state.auto_heal_enabled:
            st.markdown('<span style="color: #198754; font-weight: bold;">Self-healing ACTIVE — anomalies will be auto-corrected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #6c757d;">Self-healing disabled — monitoring only</span>', unsafe_allow_html=True)


def _render_agent_activity_banner(status_result, healing_just_happened):
    """Show compact agent status line."""
    overall = status_result.get('overall_status', 'healthy')

    if healing_just_happened:
        icon, bg, border, msg = '⚡', '#cce5ff', '#0d6efd', 'Anomaly detected → Corrective action applied → Verifying improvement...'
    elif overall == 'healthy':
        icon, bg, border, msg = '✅', '#d4edda', '#198754', 'Network healthy — Continuous monitoring active'
    elif overall == 'warning':
        heal = 'Self-healing active' if st.session_state.auto_heal_enabled else 'Self-healing disabled'
        icon, bg, border, msg = '⚠️', '#fff3cd', '#ffc107', f'Warning detected — {heal}'
    else:
        heal = 'Self-healing active' if st.session_state.auto_heal_enabled else 'Self-healing disabled'
        icon, bg, border, msg = '🚨', '#f8d7da', '#dc3545', f'Critical issue detected — {heal}'

    st.markdown(f'<div style="background:{bg};border-left:4px solid {border};padding:0.6rem 1rem;border-radius:0 8px 8px 0;margin-bottom:0.5rem;"><b>{icon} Agent Status:</b> {msg}</div>', unsafe_allow_html=True)


def _render_active_alerts_panel():
    """Render active alerts with severity-colored cards."""
    active = st.session_state.active_alerts
    if not active:
        return

    st.subheader(f"Active Alerts ({len(active)})")

    for alert in active:
        severity = alert['severity']
        if severity == 'critical':
            border_color, bg_color, icon = '#dc3545', '#f8d7da', 'CRITICAL'
        else:
            border_color, bg_color, icon = '#ffc107', '#fff3cd', 'WARNING'

        elapsed = (datetime.now() - alert['timestamp']).total_seconds()
        elapsed_str = f"{int(elapsed)}s ago" if elapsed < 60 else f"{int(elapsed/60)}m ago"

        value_str = ""
        if alert.get('value') is not None:
            value_str = f"Current: {alert['value']:.1f}"
            if alert.get('threshold') is not None:
                value_str += f" | Threshold: {alert['threshold']}"

        st.markdown(f"""
        <div style="background: {bg_color}; border-left: 5px solid {border_color};
                    padding: 0.8rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: {border_color}; color: white; padding: 0.15rem 0.5rem;
                                 border-radius: 4px; font-size: 0.75rem; font-weight: bold;">{icon}</span>
                    <b style="margin-left: 0.5rem; color: #000;">{alert['message']}</b>
                </div>
                <span style="color: #666; font-size: 0.85rem;">{elapsed_str}</span>
            </div>
            {'<p style="color: #333; margin: 0.3rem 0 0 0; font-size: 0.9rem;">' + value_str + '</p>' if value_str else ''}
        </div>
        """, unsafe_allow_html=True)


def _render_healing_result():
    """Show the most recent auto-healing before/after comparison."""
    result = st.session_state.last_healing_result
    if not result:
        return

    st.subheader("Self-Healing Result")

    health_delta = result['after_health'] - result['before_health']
    delta_color = '#198754' if health_delta > 0 else '#dc3545' if health_delta < 0 else '#6c757d'

    st.markdown(f"""
    <div style="background: #d4edda; border: 2px solid #198754; border-radius: 10px;
                padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: #000; margin: 0 0 0.5rem 0;">Autonomous Healing Complete</h4>
        <p style="color: #333; margin: 0;">
            Health Score: {result['before_health']} &rarr;
            <b style="color: {delta_color};">{result['after_health']}</b>
            ({'+' if health_delta >= 0 else ''}{health_delta} points)
        </p>
    </div>
    """, unsafe_allow_html=True)

    before = result['before_metrics']
    after = result['after_metrics']

    col_b, col_arrow, col_a = st.columns([5, 1, 5])

    with col_b:
        st.markdown("**BEFORE**")
        st.metric("Latency", f"{before['latency_ms']:.1f} ms")
        st.metric("Throughput", f"{before['throughput_mbps']:.1f} Mbps")
        st.metric("Cell Load", f"{before['cell_load_percent']:.1f}%")

    with col_arrow:
        st.markdown("")
        st.markdown("")
        st.markdown("<h1 style='text-align:center; color:#0d6efd;'>&rarr;</h1>", unsafe_allow_html=True)

    with col_a:
        st.markdown("**AFTER**")
        lat_d = after['latency_ms'] - before['latency_ms']
        st.metric("Latency", f"{after['latency_ms']:.1f} ms", delta=f"{lat_d:+.1f} ms", delta_color="inverse")
        tp_d = after['throughput_mbps'] - before['throughput_mbps']
        st.metric("Throughput", f"{after['throughput_mbps']:.1f} Mbps", delta=f"{tp_d:+.1f} Mbps")
        cl_d = after['cell_load_percent'] - before['cell_load_percent']
        st.metric("Cell Load", f"{after['cell_load_percent']:.1f}%", delta=f"{cl_d:+.1f}%", delta_color="inverse")

    for action_info in result['actions_taken']:
        success = action_info['result'].get('success', False)
        action_label = action_info['action'].replace('_', ' ').title()
        st.markdown(f"""
        <div style="background: {'#d4edda' if success else '#f8d7da'};
                    border-left: 4px solid {'#198754' if success else '#dc3545'};
                    padding: 0.6rem; margin-top: 0.5rem; border-radius: 0 5px 5px 0;">
            <b>Action:</b> {action_label} |
            <b>Reason:</b> {action_info.get('reason', 'N/A')} |
            <b>Result:</b> {'Success' if success else 'Failed'}
        </div>
        """, unsafe_allow_html=True)

    # ═══ Level 5: Show Rollback Info if applicable ═══
    rollback = result.get('rollback', {})
    if rollback.get('needed'):
        st.markdown(f"""
        <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px;
                    padding: 1rem; margin-top: 1rem;">
            <h4 style="color: #856404; margin: 0 0 0.5rem 0;">
                ⚠️ AUTO-ROLLBACK EXECUTED (Level {AUTONOMY_LEVEL})
            </h4>
            <p style="color: #856404; margin: 0 0 0.5rem 0;">
                <b>Reason:</b> {'; '.join(rollback.get('reasons', ['Metrics degraded']))}
            </p>
            <p style="color: #856404; margin: 0;">
                The system detected that the healing action made things worse and
                <b>automatically reversed</b> the changes. This is Level {AUTONOMY_LEVEL}
                self-correction in action.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif result.get('autonomy_level', 0) >= 5:
        # Show Level 5 badge for successful healing without rollback
        st.markdown(f"""
        <div style="background: #d1ecf1; border-left: 4px solid #17a2b8;
                    padding: 0.5rem; margin-top: 0.5rem; border-radius: 0 5px 5px 0;">
            <b>Level {AUTONOMY_LEVEL} Autonomous:</b> Action validated, executed, and verified —
            no rollback needed.
        </div>
        """, unsafe_allow_html=True)

    # Show skipped actions if any
    skipped = result.get('actions_skipped', [])
    if skipped:
        with st.expander(f"Skipped Actions ({len(skipped)})"):
            for skip in skipped:
                st.warning(f"**{skip['action']}** skipped: {skip['reason']}")


def _render_kpi_gauges(metrics):
    """Render KPI gauge charts for clear visual status."""
    col1, col2, col3, col4 = st.columns(4)

    # Compute deltas from previous reading
    history = st.session_state.metrics_history
    prev = history[-2] if len(history) >= 2 else None

    with col1:
        throughput = metrics['throughput_mbps']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=throughput,
            number={'suffix': ' Mbps', 'font': {'size': 22}},
            title={'text': 'Throughput', 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 200], 'ticksuffix': ''},
                'bar': {'color': '#0d6efd'},
                'steps': [
                    {'range': [0, 30], 'color': '#f8d7da'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 200], 'color': '#d4edda'},
                ],
                'threshold': {'line': {'color': '#198754', 'width': 3}, 'thickness': 0.8, 'value': KPI_THRESHOLDS['throughput_mbps']['target']},
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if prev:
            delta = throughput - prev['throughput_mbps']
            color = '#198754' if delta >= 0 else '#dc3545'
            st.markdown(f'<p style="text-align:center;margin-top:-15px;color:{color};font-weight:bold;">{delta:+.1f} Mbps</p>', unsafe_allow_html=True)

    with col2:
        latency = metrics['latency_ms']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latency,
            number={'suffix': ' ms', 'font': {'size': 22}},
            title={'text': 'Latency', 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 150], 'ticksuffix': ''},
                'bar': {'color': '#0d6efd'},
                'steps': [
                    {'range': [0, 50], 'color': '#d4edda'},
                    {'range': [50, 80], 'color': '#fff3cd'},
                    {'range': [80, 150], 'color': '#f8d7da'},
                ],
                'threshold': {'line': {'color': '#198754', 'width': 3}, 'thickness': 0.8, 'value': KPI_THRESHOLDS['latency_ms']['target']},
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if prev:
            delta = latency - prev['latency_ms']
            color = '#198754' if delta <= 0 else '#dc3545'
            st.markdown(f'<p style="text-align:center;margin-top:-15px;color:{color};font-weight:bold;">{delta:+.1f} ms</p>', unsafe_allow_html=True)

    with col3:
        cell_load = metrics['cell_load_percent']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cell_load,
            number={'suffix': '%', 'font': {'size': 22}},
            title={'text': 'Cell Load', 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': ''},
                'bar': {'color': '#0d6efd'},
                'steps': [
                    {'range': [0, 70], 'color': '#d4edda'},
                    {'range': [70, 85], 'color': '#fff3cd'},
                    {'range': [85, 100], 'color': '#f8d7da'},
                ],
                'threshold': {'line': {'color': '#198754', 'width': 3}, 'thickness': 0.8, 'value': KPI_THRESHOLDS['cell_load_percent']['target']},
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if prev:
            delta = cell_load - prev['cell_load_percent']
            color = '#198754' if delta <= 0 else '#dc3545'
            st.markdown(f'<p style="text-align:center;margin-top:-15px;color:{color};font-weight:bold;">{delta:+.1f}%</p>', unsafe_allow_html=True)

    with col4:
        users = metrics['connected_users']
        slices = metrics['active_slices']
        st.metric("Connected Users", f"{users:,}")
        st.metric("Active Slices", slices)
        if prev:
            user_delta = users - prev['connected_users']
            color = '#198754' if user_delta >= 0 else '#dc3545'
            st.markdown(f'<p style="text-align:center;color:{color};font-weight:bold;">{user_delta:+,} users</p>', unsafe_allow_html=True)


def _render_monitor_charts():
    """Render latency and throughput/cell-load charts."""
    if len(st.session_state.metrics_history) <= 1:
        return

    col1, col2 = st.columns(2)
    timestamps = [m['timestamp'] for m in st.session_state.metrics_history]

    with col1:
        fig = go.Figure()
        latencies = [m['latency_ms'] for m in st.session_state.metrics_history]
        fig.add_trace(go.Scatter(x=timestamps, y=latencies, mode='lines+markers',
                                 name='Latency', line=dict(color='#0d6efd')))
        fig.add_hline(y=KPI_THRESHOLDS['latency_ms']['warning'], line_dash="dash",
                      line_color="orange", annotation_text="Warning")
        fig.add_hline(y=KPI_THRESHOLDS['latency_ms']['critical'], line_dash="dash",
                      line_color="red", annotation_text="Critical")
        fig.update_layout(title="Latency Over Time", xaxis_title="Time",
                          yaxis_title="Latency (ms)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        throughputs = [m['throughput_mbps'] for m in st.session_state.metrics_history]
        cell_loads = [m['cell_load_percent'] for m in st.session_state.metrics_history]
        fig.add_trace(go.Scatter(x=timestamps, y=throughputs, mode='lines+markers',
                                 name='Throughput (Mbps)', line=dict(color='#198754')))
        fig.add_trace(go.Scatter(x=timestamps, y=cell_loads, mode='lines+markers',
                                 name='Cell Load (%)', line=dict(color='#ffc107'), yaxis='y2'))
        fig.update_layout(title="Throughput & Cell Load", xaxis_title="Time",
                          yaxis_title="Throughput (Mbps)",
                          yaxis2=dict(title="Cell Load (%)", overlaying='y', side='right', range=[0, 100]),
                          height=300)
        st.plotly_chart(fig, use_container_width=True)


def _render_alert_history_log():
    """Render the expandable alert history log."""
    history = st.session_state.alert_history
    if not history:
        return

    with st.expander(f"Alert History ({len(history)} total)", expanded=False):
        total = len(history)
        resolved = sum(1 for a in history if a['resolved'])
        auto_healed = sum(1 for a in history if a.get('auto_healed', False))
        critical_count = sum(1 for a in history if a['severity'] == 'critical')

        sc = st.columns(4)
        with sc[0]:
            st.metric("Total Alerts", total)
        with sc[1]:
            st.metric("Resolved", resolved)
        with sc[2]:
            st.metric("Auto-Healed", auto_healed)
        with sc[3]:
            st.metric("Critical", critical_count)

        st.divider()

        for alert in reversed(history[-20:]):
            if alert['resolved']:
                border, bg, status_label = '#198754', '#d4edda', 'RESOLVED'
            elif alert['severity'] == 'critical':
                border, bg, status_label = '#dc3545', '#f8d7da', 'ACTIVE'
            else:
                border, bg, status_label = '#ffc107', '#fff3cd', 'ACTIVE'

            ts_str = alert['timestamp'].strftime('%H:%M:%S')
            resolution_str = ""
            if alert['resolved'] and alert.get('resolution_action'):
                action_name = alert['resolution_action'].replace('_', ' ').title()
                if alert.get('auto_healed'):
                    resolution_str = f"Auto-healed via <b>{action_name}</b>"
                elif alert['resolution_action'] == 'self_resolved':
                    resolution_str = "Self-resolved (condition cleared)"
                else:
                    resolution_str = f"Resolved via <b>{action_name}</b>"
                if alert.get('resolved_at'):
                    dur = (alert['resolved_at'] - alert['timestamp']).total_seconds()
                    resolution_str += f" in {dur:.0f}s"

            st.markdown(f"""
            <div style="background: {bg}; border-left: 4px solid {border};
                        padding: 0.5rem 0.8rem; margin-bottom: 0.3rem; border-radius: 0 5px 5px 0;">
                <span style="color: #666;">[{ts_str}]</span>
                <span style="background: {border}; color: white; padding: 0.1rem 0.4rem;
                             border-radius: 3px; font-size: 0.7rem; margin: 0 0.3rem;">{status_label}</span>
                <b style="color: #000;">{alert['message']}</b>
                {f'<br/><span style="color: #555; margin-left: 4.5rem;">{resolution_str}</span>' if resolution_str else ''}
            </div>
            """, unsafe_allow_html=True)


def render_metrics_dashboard():
    """Render the enhanced network monitoring dashboard with anomaly alerts and self-healing."""
    st.header("📊 Network Monitor")

    # ── Phase 1: Collect & Analyze ──
    metrics_result = get_metrics()
    metrics = metrics_result['metrics']
    status_result = check_status(metrics_result)

    st.session_state.metrics_history.append({'timestamp': datetime.now(), **metrics})
    if len(st.session_state.metrics_history) > 50:
        st.session_state.metrics_history = st.session_state.metrics_history[-50:]
    st.session_state.monitor_cycle_count += 1

    # ── Phase 2: Process Alerts ──
    _process_alerts(status_result)

    # ── Phase 3: Auto-Heal (if enabled and needed) ──
    healing_just_happened = False
    if (st.session_state.auto_heal_enabled
            and status_result.get('requires_action', False)):
        healing_just_happened = _execute_auto_healing(status_result, metrics)

    # ── Phase 4: Render UI ──
    _render_monitor_header(status_result)
    _render_monitor_controls()
    _render_kpi_gauges(metrics)
    _render_monitor_charts()
    st.divider()
    _render_agent_activity_banner(status_result, healing_just_happened)
    _render_active_alerts_panel()
    if st.session_state.last_healing_result:
        _render_healing_result()
    _render_alert_history_log()



### ---------------------------------------------------------------------------
### Level 5 Autonomous Network - Safety Configuration
### ---------------------------------------------------------------------------

# Autonomy Level Configuration
AUTONOMY_LEVEL = 5  # Level 5 = Full Autonomy (no human intervention)

# Allowed intent types the system knows how to handle
# Intent types are no longer restricted — the LLM generates any type dynamically.
# This set is kept only for reference; validation no longer rejects unknown types.
COMMON_INTENT_TYPES = {
    "stadium_event", "concert", "emergency", "iot_deployment",
    "optimization", "general_optimization", "healthcare",
    "transportation", "smart_factory", "video_conferencing", "gaming",
    "mass_gathering", "education", "drone_operations", "smart_agriculture",
    "public_safety", "energy_grid",
}

# =============================================================================
# HARD-CODED BOUNDARIES (Level 5 Safety - CANNOT be overridden)
# =============================================================================
# These limits are derived from physical network constraints and 3GPP specs.
# Even if LLM requests values outside these bounds, the system will REJECT.

HARD_LIMITS = {
    # ─── Bandwidth Constraints ───────────────────────────────────────────
    "max_bandwidth_mbps": 500,           # Physical network cap (hardware limit)
    "min_bandwidth_mbps": 10,            # Minimum viable allocation
    "max_bandwidth_change_mbps": 100,    # Max single adjustment (prevent oscillation)

    # ─── Latency Constraints ─────────────────────────────────────────────
    "min_latency_ms": 1,                 # Sub-1ms impossible in current RAN
    "max_latency_ms": 500,               # Above this = not 5G quality

    # ─── User/Device Constraints ─────────────────────────────────────────
    "max_expected_users": 500_000,       # Network capacity limit
    "min_expected_users": 1,
    "max_users_per_cell": 200,           # 3GPP recommended limit

    # ─── Cell/Infrastructure Constraints ─────────────────────────────────
    "max_cells": 50,                     # Physical infrastructure limit
    "max_cells_activate_at_once": 5,     # Prevent mass activation
    "min_active_cells": 5,               # Always keep minimum coverage

    # ─── Power Constraints ───────────────────────────────────────────────
    "max_power_watts": 1000,             # Regulatory/hardware limit
    "min_power_watts": 100,              # Minimum for operation

    # ─── Slice Constraints ───────────────────────────────────────────────
    "max_slices": 10,                    # Concurrent slice limit
    "max_slice_bandwidth_percent": 80,   # No single slice takes all resources
}

# =============================================================================
# CONFIDENCE THRESHOLDS (Level 5 = Higher bar for autonomous execution)
# =============================================================================
CONFIDENCE_THRESHOLDS = {
    "level_4": {
        "auto_execute": 0.70,            # Level 4: 70%+ = execute, below = ask human
        "reject": 0.30,                  # Below 30% = reject outright
    },
    "level_5": {
        "auto_execute": 0.90,            # Level 5: Only 90%+ gets executed
        "reject": 0.70,                  # Below 70% = reject (no human to ask)
    }
}

# =============================================================================
# AUTO-ROLLBACK CONFIGURATION
# =============================================================================
ROLLBACK_CONFIG = {
    "enabled": True,
    "check_delay_seconds": 10,           # Wait before checking results (was 30, reduced for demo)
    "health_drop_threshold": 10,         # Rollback if health drops by more than 10 points
    "metric_degradation_percent": 20,    # Rollback if any metric worsens by 20%+
    "max_rollback_attempts": 3,          # Don't get stuck in rollback loops
}

# =============================================================================
# ALLOWED HEALING ACTIONS (Bounded action space)
# =============================================================================
ALLOWED_HEALING_ACTIONS = {
    "scale_bandwidth": {
        "enabled": True,
        "max_change": 100,               # Max 100 Mbps per action
        "cooldown_seconds": 60,          # Wait between same actions
    },
    "activate_cell": {
        "enabled": True,
        "max_count": 5,                  # Max 5 cells at once
        "cooldown_seconds": 120,
    },
    "modify_qos": {
        "enabled": True,
        "allowed_changes": ["priority", "latency_target"],
        "cooldown_seconds": 30,
    },
    "energy_saving": {
        "enabled": True,
        "min_load_percent": 30,          # Only if load < 30%
        "cooldown_seconds": 300,
    },
    "adjust_priority": {
        "enabled": True,
        "max_priority_change": 2,        # Don't jump more than 2 levels
        "cooldown_seconds": 60,
    },
}

# Legacy alias for backward compatibility
SAFETY_RULES = {
    "max_bandwidth_mbps": HARD_LIMITS["max_bandwidth_mbps"],
    "min_latency_ms": HARD_LIMITS["min_latency_ms"],
    "max_latency_ms": HARD_LIMITS["max_latency_ms"],
    "max_expected_users": HARD_LIMITS["max_expected_users"],
    "min_expected_users": HARD_LIMITS["min_expected_users"],
    "allowed_priorities": {"critical", "high", "normal", "low"},
    "max_priority_numeric": 10,
    "min_priority_numeric": 1,
    "min_confidence": CONFIDENCE_THRESHOLDS[f"level_{AUTONOMY_LEVEL}"]["reject"],
}


def validate_intent(parsed_intent: dict) -> dict:
    """
    Rule-based Intent Validator (Agent 2) - Level 5 Autonomous Safety Gate.

    Checks the output of the Intent Interpreter against deterministic safety
    rules BEFORE any configuration is generated.  This ensures the LLM acts
    only as a translator — actual control passes through validated rules.

    Level 5 Enhancements:
    - Higher confidence threshold (90% for auto-execution)
    - Hard limits validation against physical network constraints
    - Detailed audit logging of validation decisions

    Returns:
        {
            "approved": bool,
            "risk_level": "safe" | "caution" | "blocked",
            "checks_passed": [...],
            "warnings": [...],
            "rejections": [...],
            "validated_intent": <cleaned intent dict or None>,
            "autonomy_level": int,
            "confidence_details": {...},
        }
    """
    checks_passed = []
    warnings = []
    rejections = []

    # Get confidence thresholds for current autonomy level
    level_key = f"level_{AUTONOMY_LEVEL}"
    conf_thresholds = CONFIDENCE_THRESHOLDS.get(level_key, CONFIDENCE_THRESHOLDS["level_5"])

    # --- 1. Parse success check ------------------------------------------
    if not parsed_intent.get("parsed_successfully"):
        rejections.append("Intent parsing failed — cannot validate.")
        return _build_validation_result(False, "blocked", checks_passed, warnings, rejections, None, 0, conf_thresholds)
    checks_passed.append("Intent parsed successfully")

    intent_type = parsed_intent.get("intent_type", "unknown")
    entities = parsed_intent.get("entities", {})
    confidence = parsed_intent.get("confidence", 0)

    # --- 2. Intent type check (any LLM-generated type is accepted) --------
    if not intent_type or intent_type == "unknown":
        rejections.append("Intent type is missing or unknown — cannot validate.")
    else:
        checks_passed.append(f"Intent type '{intent_type}' is recognized")

    # --- 3. LEVEL 5: Strict Confidence threshold -------------------------
    auto_execute_threshold = conf_thresholds["auto_execute"]
    reject_threshold = conf_thresholds["reject"]

    if confidence < reject_threshold:
        rejections.append(
            f"[Level {AUTONOMY_LEVEL}] Parser confidence {confidence:.0%} is below "
            f"minimum {reject_threshold:.0%} — intent too ambiguous for autonomous execution."
        )
    elif confidence < auto_execute_threshold:
        # In Level 5, we can't ask human, so this becomes a warning but still executes
        # with extra caution (bounded actions only)
        warnings.append(
            f"[Level {AUTONOMY_LEVEL}] Confidence {confidence:.0%} is below optimal "
            f"{auto_execute_threshold:.0%} — executing with extra safety bounds."
        )
        checks_passed.append(f"Confidence {confidence:.0%} acceptable (with caution)")
    else:
        checks_passed.append(f"Confidence {confidence:.0%} meets Level {AUTONOMY_LEVEL} threshold ({auto_execute_threshold:.0%})")

    # --- 4. Expected users vs HARD LIMITS --------------------------------
    expected_users = entities.get("expected_users", 10_000)
    max_users = HARD_LIMITS["max_expected_users"]
    min_users = HARD_LIMITS["min_expected_users"]

    if expected_users > max_users:
        rejections.append(
            f"[HARD LIMIT] Expected users ({expected_users:,}) exceeds "
            f"physical network capacity ({max_users:,})."
        )
    elif expected_users < min_users:
        rejections.append(f"[HARD LIMIT] Expected users must be at least {min_users}.")
    else:
        checks_passed.append(f"Expected users ({expected_users:,}) within hard limits")

    # --- 5. Priority validation ------------------------------------------
    priority = entities.get("priority", "normal")
    if priority not in SAFETY_RULES["allowed_priorities"]:
        warnings.append(f"Priority '{priority}' not recognized — defaulting to 'normal'.")
        entities["priority"] = "normal"
    else:
        checks_passed.append(f"Priority '{priority}' is valid")

    # --- 6. Recommended profile vs HARD LIMITS ---------------------------
    profile = parsed_intent.get("recommended_event_profile", {})
    bw_mult = profile.get("bandwidth_multiplier", 1.0)

    # Calculate effective bandwidth request
    effective_bw = bw_mult * 100  # Base bandwidth assumption
    if effective_bw > HARD_LIMITS["max_bandwidth_mbps"]:
        warnings.append(
            f"[HARD LIMIT] Effective bandwidth ({effective_bw:.0f} Mbps) may exceed "
            f"max ({HARD_LIMITS['max_bandwidth_mbps']} Mbps) — will be capped."
        )
    elif bw_mult > 5.0:
        warnings.append(
            f"Bandwidth multiplier ({bw_mult}x) is very high — may overload network."
        )
    else:
        checks_passed.append(f"Bandwidth multiplier ({bw_mult}x) is reasonable")

    extra_cells = profile.get("additional_cells", 0)
    max_cells = HARD_LIMITS["max_cells"]
    if extra_cells > max_cells:
        rejections.append(
            f"[HARD LIMIT] Requested {extra_cells} additional cells exceeds "
            f"physical infrastructure limit ({max_cells})."
        )
    elif extra_cells > HARD_LIMITS["max_cells_activate_at_once"]:
        warnings.append(
            f"Requested {extra_cells} cells exceeds safe activation limit "
            f"({HARD_LIMITS['max_cells_activate_at_once']}) — will be staged."
        )
        checks_passed.append("Cell allocation will be staged for safety")
    else:
        checks_passed.append("Cell allocation within limits")

    # --- 7. Emergency intent requires critical priority ------------------
    if intent_type == "emergency" and priority != "critical":
        warnings.append(
            "Emergency intent detected but priority is not 'critical' — "
            "escalating to critical automatically."
        )
        entities["priority"] = "critical"

    # --- 8. Additional Level 5 Hard Limit Checks -------------------------
    # Check for any dangerous parameter combinations
    if intent_type == "emergency" and expected_users > 100_000:
        warnings.append(
            "Emergency + high user count — network resources will be prioritized "
            "for emergency services first."
        )

    # --- Build final result ----------------------------------------------
    approved = len(rejections) == 0
    if not approved:
        risk_level = "blocked"
    elif len(warnings) > 0:
        risk_level = "caution"
    else:
        risk_level = "safe"

    validated_intent = parsed_intent.copy() if approved else None
    if validated_intent:
        validated_intent["entities"] = entities  # may have been patched
        validated_intent["validation"] = {
            "approved": True,
            "risk_level": risk_level,
            "warnings": warnings,
            "autonomy_level": AUTONOMY_LEVEL,
        }

    return _build_validation_result(
        approved, risk_level, checks_passed, warnings, rejections,
        validated_intent, confidence, conf_thresholds
    )


def _build_validation_result(approved, risk_level, checks_passed, warnings, rejections,
                              validated_intent, confidence, conf_thresholds):
    """Build validation result with Level 5 autonomy details."""
    return {
        "approved": approved,
        "risk_level": risk_level,
        "checks_passed": checks_passed,
        "checks_total": len(checks_passed) + len(warnings) + len(rejections),
        "warnings": warnings,
        "rejections": rejections,
        "validated_intent": validated_intent,
        "autonomy_level": AUTONOMY_LEVEL,
        "confidence_details": {
            "actual": confidence,
            "auto_execute_threshold": conf_thresholds["auto_execute"],
            "reject_threshold": conf_thresholds["reject"],
            "meets_auto_execute": confidence >= conf_thresholds["auto_execute"],
        },
        "hard_limits_applied": list(HARD_LIMITS.keys()),
    }


# Legacy function for backward compatibility
def _build_result(approved, risk_level, checks_passed, warnings, rejections, validated_intent):
    return _build_validation_result(
        approved, risk_level, checks_passed, warnings, rejections,
        validated_intent, 0, CONFIDENCE_THRESHOLDS[f"level_{AUTONOMY_LEVEL}"]
    )


def run_agents_phase1(user_intent: str):
    """Phase 1: Run Agent 1 (Intent Interpreter) + Reasoner question generation."""

    # Reset agent states
    for agent in st.session_state.agent_states:
        st.session_state.agent_states[agent] = {'status': 'waiting', 'output': None}
    st.session_state.reasoning_phase = None
    st.session_state.current_config = None

    progress_bar = st.progress(0, text="Starting agent pipeline...")

    # ── Agent 1: Intent Interpreter ──────────────────────────────────────
    st.session_state.agent_states['intent']['status'] = 'running'
    with st.status("🧠 Agent 1: Intent Interpreter — Analyzing request...", expanded=True) as status1:
        st.write("Parsing natural language input...")
        time.sleep(0.5)
        st.write("Detecting intent type and extracting entities...")
        time.sleep(0.5)

        try:
            intent_result = get_optimization_crew().run_intent_for_app(user_intent)
            st.session_state.agent_states['intent']['status'] = 'completed'
            st.session_state.agent_states['intent']['output'] = intent_result

            intent_type = intent_result.get('intent_type', 'unknown')
            confidence = intent_result.get('confidence', 0)
            app_type = intent_result.get('entities', {}).get('application', 'mixed')
            # st.write(f"**Intent:** {intent_type} | **Confidence:** {confidence:.0%} | **Application:** {app_type}")
            llm_badge = "🤖 AI-Powered" if intent_result.get("llm_powered") else "⚙️ Keyword Fallback"
            st.write(f"**Intent:** {intent_type} | **Confidence:** {confidence:.0%} | **Application:** {app_type} | {llm_badge}")
            status1.update(label=f"🧠 Agent 1: Intent Interpreter — {intent_type} (confidence: {confidence:.0%})", state="complete", expanded=False)
        except Exception as e:
            st.session_state.agent_states['intent']['status'] = 'error'
            st.session_state.agent_states['intent']['output'] = {'error': str(e)}
            st.error(f"Error: {e}")
            status1.update(label="🧠 Agent 1: Intent Interpreter — Error", state="error", expanded=False)
            return

    progress_bar.progress(17, text="1/6 Agents Complete")

    # ── Agent 2: Reasoner — Generate clarifying questions ────────────────
    st.session_state.agent_states['reasoner']['status'] = 'running'
    with st.status("🤔 Agent 2: Reasoner — Generating clarifying questions...", expanded=True) as status2:
        st.write("Analyzing intent context...")
        time.sleep(0.5)
        st.write("Preparing scenario-specific questions...")
        time.sleep(0.5)

        questions = _generate_reasoning_questions(intent_result)
        st.session_state.reasoning_questions = questions
        st.session_state.partial_intent_result = intent_result

        st.write(f"**Generated {len(questions)} clarifying questions** for intent: {intent_type}")
        status2.update(label=f"🤔 Agent 2: Reasoner — Awaiting user input ({len(questions)} questions)", state="running", expanded=False)

    st.session_state.reasoning_phase = 'questions_ready'
    progress_bar.progress(25, text="Awaiting user confirmation...")


def run_agents_phase2():
    """Phase 2: Run Reasoner analysis + Agents 3-6 after user confirms."""

    intent_result = st.session_state.partial_intent_result
    answers = st.session_state.reasoning_answers

    progress_bar = st.progress(25, text="Resuming agent pipeline...")

    # ── Agent 2: Reasoner — Feasibility, Risk, Impact ────────────────────
    with st.status("🤔 Agent 2: Reasoner — Analyzing feasibility, risks, and impact...", expanded=True) as status2:
        st.write("Groq LLM analysing intent feasibility and risks...")
        time.sleep(0.5)

        metrics_result = get_metrics()
        metrics = metrics_result['metrics']

        # Route through the planner agent (Groq LLM) for intelligent reasoning
        llm_reasoning = get_optimization_crew().run_reasoner_for_app(intent_result, answers, metrics)

        # Use LLM result if valid, otherwise fall back to rule-based functions
        feasibility = llm_reasoning.get('feasibility') if llm_reasoning else None
        if feasibility is None:
            feasibility = _check_feasibility(intent_result, answers, metrics)

        risks = llm_reasoning.get('risks') if llm_reasoning else None
        if risks is None:
            risks = _identify_risks(intent_result, answers, metrics)

        impact = llm_reasoning.get('impact') if llm_reasoning else None
        if impact is None:
            impact = _simulate_impact(intent_result, answers, metrics)

        high_risks = sum(1 for r in risks if r.get('severity') == 'HIGH')
        before_lat = impact['before']['latency_ms']
        after_lat = impact['predicted_after']['latency_ms']

        st.write(f"**Feasibility:** {feasibility['feasibility_score']}/100" + (" — Feasible" if feasibility['feasible'] else " — Constrained"))
        st.write(f"**Risks found:** {len(risks)} ({high_risks} high severity)")
        st.write(f"**Predicted latency:** {before_lat:.0f}ms → {after_lat:.0f}ms")

        reasoning_output = {
            'answers': answers,
            'feasibility': feasibility,
            'risks': risks,
            'simulated_impact': impact,
        }
        st.session_state.agent_states['reasoner']['status'] = 'completed'
        st.session_state.agent_states['reasoner']['output'] = reasoning_output

        status2.update(label=f"🤔 Agent 2: Reasoner — Feasibility {feasibility['feasibility_score']}/100, {len(risks)} risks", state="complete", expanded=False)

    progress_bar.progress(33, text="2/6 Agents Complete")

    # ── Agent 3: Intent Validator ────────────────────────────────────────
    st.session_state.agent_states['validator']['status'] = 'running'
    with st.status("🛡️ Agent 3: Intent Validator — Checking safety rules...", expanded=True) as status3:
        st.write("Validating against safety constraints...")
        time.sleep(0.5)
        st.write("Checking resource limits and policy compliance...")
        time.sleep(0.5)

        try:
            validation_result = validate_intent(intent_result)
            st.session_state.agent_states['validator']['status'] = 'completed'
            st.session_state.agent_states['validator']['output'] = validation_result

            risk = validation_result.get('risk_level', 'UNKNOWN')
            checks = validation_result.get('checks_passed', 0)

            if not validation_result["approved"]:
                rejections = '; '.join(validation_result.get('rejections', []))
                st.error(f"**BLOCKED** — {rejections}")
                status3.update(label="🛡️ Agent 3: Intent Validator — BLOCKED", state="error", expanded=True)
                progress_bar.progress(50, text="Pipeline stopped — Intent rejected")
                return

            st.write(f"**Risk:** {risk} | **Checks passed:** {checks}")
            status3.update(label=f"🛡️ Agent 3: Intent Validator — Approved (Risk: {risk})", state="complete", expanded=False)
            intent_result = validation_result["validated_intent"]
        except Exception as e:
            st.session_state.agent_states['validator']['status'] = 'error'
            st.session_state.agent_states['validator']['output'] = {'error': str(e)}
            st.error(f"Error: {e}")
            status3.update(label="🛡️ Agent 3: Intent Validator — Error", state="error", expanded=False)
            return

    progress_bar.progress(50, text="3/6 Agents Complete")

    # ── Agent 4: Planner ─────────────────────────────────────────────────
    st.session_state.agent_states['planner']['status'] = 'running'
    with st.status("📋 Agent 4: Planner — Generating configuration...", expanded=True) as status4:
        st.write("Selecting optimal network slice...")
        time.sleep(0.5)
        st.write("Computing resource allocation parameters...")
        time.sleep(0.5)

        try:
            config_result = get_optimization_crew().run_planner_for_app(intent_result)
            st.session_state.agent_states['planner']['status'] = 'completed'
            st.session_state.agent_states['planner']['output'] = config_result
            st.session_state.current_config = config_result

            if 'network_slice' in config_result:
                slice_config = config_result['network_slice']
                network_simulator.create_slice(
                    slice_type=slice_config['type'],
                    bandwidth_mbps=slice_config['allocated_bandwidth_mbps'],
                    latency_target_ms=slice_config['latency_target_ms'],
                    priority=slice_config['priority']
                )
                st.write(f"**Slice:** {slice_config['type']} | **BW:** {slice_config['allocated_bandwidth_mbps']} Mbps | **Latency:** {slice_config['latency_target_ms']} ms | **Priority:** {slice_config['priority']}")
                status4.update(label=f"📋 Agent 4: Planner — {slice_config['type']} slice configured", state="complete", expanded=False)
            else:
                status4.update(label="📋 Agent 4: Planner — Configuration generated", state="complete", expanded=False)
        except Exception as e:
            st.session_state.agent_states['planner']['status'] = 'error'
            st.session_state.agent_states['planner']['output'] = {'error': str(e)}
            st.error(f"Error: {e}")
            status4.update(label="📋 Agent 4: Planner — Error", state="error", expanded=False)
            return

    progress_bar.progress(67, text="4/6 Agents Complete")

    # ── Agent 5: Monitor ─────────────────────────────────────────────────
    st.session_state.agent_states['monitor']['status'] = 'running'
    with st.status("📊 Agent 5: Monitor — Analyzing network status...", expanded=True) as status5:
        st.write("Collecting real-time network metrics...")
        time.sleep(0.5)
        st.write("Evaluating health indicators and anomalies...")
        time.sleep(0.5)

        try:
            metrics_result, status_result = get_optimization_crew().run_monitoring_for_app()
            st.session_state.agent_states['monitor']['status'] = 'completed'
            st.session_state.agent_states['monitor']['output'] = status_result

            health = status_result.get('overall_status', 'unknown')
            avg_lat = status_result.get('metrics', {}).get('avg_latency_ms', 0)
            avg_load = status_result.get('metrics', {}).get('avg_cell_load', 0)
            st.write(f"**Health:** {health} | **Avg Latency:** {avg_lat:.1f} ms | **Avg Load:** {avg_load:.1f}%")
            needs_action = status_result.get('requires_action', False)
            status5.update(label=f"📊 Agent 5: Monitor — Network {health}" + (" (action needed)" if needs_action else ""), state="complete", expanded=False)
        except Exception as e:
            st.session_state.agent_states['monitor']['status'] = 'error'
            st.session_state.agent_states['monitor']['output'] = {'error': str(e)}
            st.error(f"Error: {e}")
            status5.update(label="📊 Agent 5: Monitor — Error", state="error", expanded=False)
            return

    progress_bar.progress(83, text="5/6 Agents Complete")

    # ── Agent 6: Optimizer ───────────────────────────────────────────────
    st.session_state.agent_states['optimizer']['status'] = 'running'
    if status_result.get('requires_action', False):
        with st.status("⚡ Agent 6: Optimizer — Executing optimization...", expanded=True) as status6:
            st.write("Analyzing recommendations...")
            time.sleep(0.5)

            try:
                st.write("Groq LLM selecting optimal RAN action and parameters...")
                time.sleep(0.5)

                opt_result = get_optimization_crew().run_optimizer_for_app(status_result)
                action = opt_result.get('action', 'optimization')

                st.session_state.agent_states['optimizer']['status'] = 'completed'
                st.session_state.agent_states['optimizer']['output'] = opt_result

                st.session_state.optimization_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': action,
                    'result': opt_result
                })
                st.write(f"**Action:** {action} | **Status:** Applied successfully")
                status6.update(label=f"⚡ Agent 6: Optimizer — Applied {action}", state="complete", expanded=False)
            except Exception as e:
                st.session_state.agent_states['optimizer']['status'] = 'error'
                st.session_state.agent_states['optimizer']['output'] = {'error': str(e)}
                st.error(f"Error: {e}")
                status6.update(label="⚡ Agent 6: Optimizer — Error", state="error", expanded=False)
    else:
        with st.status("⚡ Agent 6: Optimizer — Checking if optimization needed...", expanded=True) as status6:
            st.write("Evaluating network state...")
            time.sleep(0.5)
            st.session_state.agent_states['optimizer']['status'] = 'completed'
            st.session_state.agent_states['optimizer']['output'] = {
                'message': 'No optimization needed - network is healthy'
            }
            st.write("**Status:** Network is healthy — no optimization needed")
            status6.update(label="⚡ Agent 6: Optimizer — No action needed", state="complete", expanded=False)

    progress_bar.progress(100, text="All 6 agents completed successfully!")
    st.session_state.reasoning_phase = None


def render_configuration_details():
    """Render the current configuration details"""
    if st.session_state.current_config:
        st.header("📋 Current Configuration")

        config = st.session_state.current_config

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Network Slice")
            if 'network_slice' in config:
                slice_info = config['network_slice']
                st.write(f"**Type:** {slice_info.get('type', 'N/A')}")
                st.write(f"**Name:** {slice_info.get('name', 'N/A')}")
                st.write(f"**SST:** {slice_info.get('sst', 'N/A')}")
                st.write(f"**Bandwidth:** {slice_info.get('allocated_bandwidth_mbps', 'N/A')} Mbps")
                st.write(f"**Latency Target:** {slice_info.get('latency_target_ms', 'N/A')} ms")

        with col2:
            st.subheader("QoS Parameters")
            if 'qos_parameters' in config:
                qos = config['qos_parameters']
                st.write(f"**5QI:** {qos.get('5qi', 'N/A')}")
                st.write(f"**Priority:** {qos.get('arp_priority', 'N/A')}")
                st.write(f"**Delay Budget:** {qos.get('packet_delay_budget_ms', 'N/A')} ms")
                st.write(f"**Error Rate:** {qos.get('packet_error_rate', 'N/A')}")

        with col3:
            st.subheader("RAN Configuration")
            if 'ran_configuration' in config:
                ran = config['ran_configuration']
                st.write(f"**Active Cells:** {ran.get('active_cells', 'N/A')}")
                st.write(f"**MIMO:** {ran.get('mimo_configuration', 'N/A')}")
                st.write(f"**Scheduler:** {ran.get('scheduler_type', 'N/A')}")
                st.write(f"**Spectrum:** {ran.get('spectrum_allocation_mhz', 'N/A')} MHz")


def render_optimization_log():
    """Render the optimization history log"""
    if st.session_state.optimization_log:
        st.header("📜 Optimization Log")

        for i, log in enumerate(reversed(st.session_state.optimization_log[-5:])):
            with st.expander(f"Optimization {len(st.session_state.optimization_log) - i}: {log['action']}"):
                st.write(f"**Timestamp:** {log['timestamp']}")
                st.write(f"**Action:** {log['action']}")
                if 'result' in log:
                    result = log['result']
                    if 'execution_details' in result:
                        details = result['execution_details']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Before:**")
                            st.json(details.get('before', {}))
                        with col2:
                            st.write("**After:**")
                            st.json(details.get('after', {}))


# ==========================================================================
# MULTI-INTENT CONFLICT RESOLUTION
# ==========================================================================

TOTAL_AVAILABLE_BANDWIDTH_MBPS = 500  # From 400MHz spectrum
TOTAL_AVAILABLE_CELLS = NETWORK_CONFIG["max_cells"]

STAKEHOLDER_ICONS = {
    "healthcare": "🏥", "emergency": "🏥",
    "stadium_event": "🏟️", "concert": "🎵",
    "smart_factory": "🏭", "iot_deployment": "📡",
    "transportation": "🚗", "gaming": "🎮",
    "video_conferencing": "💻",
    "general_optimization": "📊", "optimization": "📊",
}

STAKEHOLDER_LABELS = {
    "healthcare": "Hospital", "emergency": "Hospital",
    "stadium_event": "Stadium", "concert": "Concert Venue",
    "smart_factory": "Smart Factory", "iot_deployment": "IoT Network",
    "transportation": "Transportation", "gaming": "Gaming Arena",
    "video_conferencing": "Enterprise VC",
    "general_optimization": "General", "optimization": "General",
}


# _get_adjustment_suggestion removed — now handled by resolve_conflicts_with_llm()


def run_conflict_resolution(intent_texts):
    """Run the full multi-intent conflict resolution pipeline with agent negotiation"""

    negotiation_log = []

    # ── Step 1: Parse all intents ──
    step_placeholder = st.empty()
    with step_placeholder.container():
        st.markdown("""
        <div style="background-color: #cce5ff; border: 2px solid #0d6efd; border-radius: 10px; padding: 1rem;">
            <b>🔄 Agent 1 — Intent Interpreter:</b> Analyzing all stakeholder requests...
        </div>
        """, unsafe_allow_html=True)
    time.sleep(1.5)

    parsed_intents = []
    for text in intent_texts:
        parsed = parse_intent(text)
        parsed_intents.append(parsed)

    negotiation_log.append({
        "agent": "Intent Interpreter",
        "message": f"Identified {len(parsed_intents)} stakeholder requests: " +
                   ", ".join([p['intent_type'].replace('_', ' ').title() for p in parsed_intents])
    })
    step_placeholder.empty()

    # ── Step 2: Generate configs ──
    step_placeholder = st.empty()
    with step_placeholder.container():
        st.markdown("""
        <div style="background-color: #cce5ff; border: 2px solid #0d6efd; border-radius: 10px; padding: 1rem;">
            <b>🔄 Agent 2 — Planner:</b> Generating 3GPP Release 18 configurations for each stakeholder...
        </div>
        """, unsafe_allow_html=True)
    time.sleep(1.5)

    configs = []
    for parsed in parsed_intents:
        config = generate_config(parsed)
        intent_type = parsed["intent_type"]
        configs.append({
            "intent": parsed,
            "config": config,
            "intent_type": intent_type,
            "label": STAKEHOLDER_LABELS.get(intent_type, intent_type.replace("_", " ").title()),
            "icon": STAKEHOLDER_ICONS.get(intent_type, "📊"),
            "slice_type": config["network_slice"]["type"],
            "bandwidth_requested": config["network_slice"]["allocated_bandwidth_mbps"],
            "cells_requested": config["ran_configuration"]["active_cells"],
            "latency_target": config["network_slice"]["latency_target_ms"],
            "priority": config["network_slice"]["priority"],
            "users": parsed["entities"]["expected_users"],
        })

    negotiation_log.append({
        "agent": "Planner",
        "message": "Generated configurations: " +
                   ", ".join([f"{c['icon']}{c['label']} needs {c['bandwidth_requested']:.0f} Mbps ({c['slice_type']})" for c in configs])
    })
    step_placeholder.empty()

    # ── Step 3: Detect conflicts ──
    step_placeholder = st.empty()
    with step_placeholder.container():
        st.markdown("""
        <div style="background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 1rem;">
            <b>🔄 Agent 3 — Monitor:</b> Checking resource availability and detecting conflicts...
        </div>
        """, unsafe_allow_html=True)
    time.sleep(1.5)

    total_bw = sum(c["bandwidth_requested"] for c in configs)
    total_cells = sum(c["cells_requested"] for c in configs)
    bw_conflict = total_bw > TOTAL_AVAILABLE_BANDWIDTH_MBPS
    cell_conflict = total_cells > TOTAL_AVAILABLE_CELLS
    has_conflict = bw_conflict or cell_conflict

    if has_conflict:
        negotiation_log.append({
            "agent": "Monitor",
            "message": f"CONFLICT DETECTED — Demand ({total_bw:.0f} Mbps, {total_cells} cells) "
                       f"exceeds capacity ({TOTAL_AVAILABLE_BANDWIDTH_MBPS} Mbps, {TOTAL_AVAILABLE_CELLS} cells). "
                       f"Overload: {((total_bw / TOTAL_AVAILABLE_BANDWIDTH_MBPS) - 1) * 100:.0f}% bandwidth, "
                       f"{((total_cells / TOTAL_AVAILABLE_CELLS) - 1) * 100:.0f}% cells."
        })
    else:
        negotiation_log.append({
            "agent": "Monitor",
            "message": "No conflict — all demands can be accommodated within available resources."
        })
    step_placeholder.empty()

    # ── Step 4: LLM-powered conflict resolution ──
    step_placeholder = st.empty()
    with step_placeholder.container():
        st.markdown("""
        <div style="background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 10px; padding: 1rem;">
            <b>⚡ Agent 4 — Optimizer:</b> Negotiating resource allocation with AI...
        </div>
        """, unsafe_allow_html=True)
    time.sleep(1.5)
    step_placeholder.empty()

    llm_result = resolve_conflicts_with_llm(
        configs, TOTAL_AVAILABLE_BANDWIDTH_MBPS, TOTAL_AVAILABLE_CELLS
    )

    # Map LLM allocations back to the full config dicts
    resolutions = []
    resolved_types = set()
    for alloc in llm_result["allocations"]:
        cfg = next((c for c in configs if c["intent_type"] == alloc["intent_type"]), None)
        if cfg is None:
            continue
        resolved_types.add(alloc["intent_type"])
        resolutions.append({
            **cfg,
            "bandwidth_allocated": alloc["allocated_bandwidth_mbps"],
            "cells_allocated":     alloc["allocated_cells"],
            "satisfaction":        alloc["satisfaction_score"],
            "adjustment":          alloc["adjustment_suggestion"],
        })
    # Safety net: add any configs the LLM missed
    for cfg in configs:
        if cfg["intent_type"] not in resolved_types:
            resolutions.append({
                **cfg,
                "bandwidth_allocated": 0,
                "cells_allocated":     0,
                "satisfaction":        0,
                "adjustment":          "No resources could be allocated.",
            })

    negotiation_log.append({
        "agent": "Optimizer",
        "message": llm_result.get(
            "negotiation_narrative",
            "AI-powered resolution applied: " +
            " → ".join([f"{r['icon']}{r['label']} gets {r['satisfaction']}%" for r in resolutions])
        ),
    })

    # Save to session state
    st.session_state.conflict_results = {
        "configs": configs,
        "resolutions": resolutions,
        "total_bw_requested": total_bw,
        "total_cells_requested": total_cells,
        "available_bw": TOTAL_AVAILABLE_BANDWIDTH_MBPS,
        "available_cells": TOTAL_AVAILABLE_CELLS,
        "has_conflict": has_conflict,
        "bw_conflict": bw_conflict,
        "cell_conflict": cell_conflict,
        "negotiation_log": negotiation_log,
    }

    st.success("All agents completed conflict resolution!")


def render_conflict_results():
    """Render the conflict resolution results with visualizations"""
    if not st.session_state.conflict_results:
        return

    results = st.session_state.conflict_results
    resolutions = results["resolutions"]
    negotiation_log = results.get("negotiation_log", [])

    # ── Agent Negotiation Log ──
    st.subheader("🤖 Agent Negotiation Process")

    for entry in negotiation_log:
        agent = entry["agent"]
        msg = entry["message"]

        if "CONFLICT" in msg:
            color, border = "#fff3cd", "#ffc107"
        elif "No conflict" in msg:
            color, border = "#d4edda", "#198754"
        elif "Resolution" in msg or "resolution" in msg:
            color, border = "#d4edda", "#198754"
        else:
            color, border = "#e8f4fd", "#0d6efd"

        st.markdown(f"""
        <div style="background-color: {color}; border-left: 4px solid {border};
                    padding: 0.8rem; margin-bottom: 0.5rem; border-radius: 0 5px 5px 0;">
            <b>🤖 {agent}:</b> {msg}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Resource Demand vs Capacity ──
    st.subheader("📋 Resource Demand vs Capacity")

    col1, col2 = st.columns(2)

    with col1:
        bw_ratio = min(1.0, results["total_bw_requested"] / results["available_bw"])
        st.markdown(f"**Bandwidth:** {results['total_bw_requested']:.0f} / {results['available_bw']} Mbps")
        st.progress(bw_ratio)
        if results["bw_conflict"]:
            shortage = results["total_bw_requested"] - results["available_bw"]
            st.error(f"Shortage: {shortage:.0f} Mbps over capacity")

    with col2:
        cell_ratio = min(1.0, results["total_cells_requested"] / results["available_cells"])
        st.markdown(f"**Cells:** {results['total_cells_requested']} / {results['available_cells']} cells")
        st.progress(cell_ratio)
        if results["cell_conflict"]:
            shortage = results["total_cells_requested"] - results["available_cells"]
            st.error(f"Shortage: {shortage} cells over capacity")

    st.divider()

    # ── Per-Stakeholder Resolution Cards ──
    st.subheader("⚡ Priority-Based Resolution")

    sorted_res = sorted(resolutions, key=lambda x: x["priority"])

    for res in sorted_res:
        sat = res["satisfaction"]
        if sat >= 90:
            bar_color, status_text = "#198754", "Full Allocation"
        elif sat >= 60:
            bar_color, status_text = "#ffc107", "Reduced Allocation"
        else:
            bar_color, status_text = "#dc3545", "Severely Limited"

        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 5px solid {bar_color};
                    padding: 1rem; margin-bottom: 0.8rem; border-radius: 0 8px 8px 0;">
            <h4 style="color: #000; margin: 0 0 0.5rem 0;">
                {res['icon']} {res['label']} — {status_text} ({sat}%)
            </h4>
            <table style="color: #333; width: 100%;">
                <tr>
                    <td><b>Slice:</b> {res['slice_type']}</td>
                    <td><b>Priority:</b> {res['priority']}</td>
                    <td><b>Users:</b> {res['users']:,}</td>
                </tr>
                <tr>
                    <td><b>Bandwidth:</b> {res['bandwidth_allocated']:.0f} / {res['bandwidth_requested']:.0f} Mbps</td>
                    <td><b>Cells:</b> {res['cells_allocated']} / {res['cells_requested']}</td>
                    <td><b>Latency:</b> {res['latency_target']:.0f} ms</td>
                </tr>
            </table>
            <p style="color: #555; margin: 0.5rem 0 0 0;">
                <b>AI Adjustment:</b> {res['adjustment']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Bandwidth Chart ──
    st.subheader("📊 Resource Allocation Overview")

    fig = go.Figure()

    labels = [f"{r['icon']} {r['label']}" for r in resolutions]
    requested = [r["bandwidth_requested"] for r in resolutions]
    allocated = [r["bandwidth_allocated"] for r in resolutions]

    fig.add_trace(go.Bar(
        name="Requested",
        x=labels,
        y=requested,
        marker_color="rgba(13, 110, 253, 0.3)",
        text=[f"{v:.0f} Mbps" for v in requested],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="Allocated",
        x=labels,
        y=allocated,
        marker_color="rgba(13, 110, 253, 1.0)",
        text=[f"{v:.0f} Mbps" for v in allocated],
        textposition="auto"
    ))

    fig.add_hline(
        y=TOTAL_AVAILABLE_BANDWIDTH_MBPS,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Total Capacity: {TOTAL_AVAILABLE_BANDWIDTH_MBPS} Mbps"
    )

    fig.update_layout(
        title="Bandwidth: Requested vs Allocated (Mbps)",
        barmode="group",
        yaxis_title="Bandwidth (Mbps)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Satisfaction Summary ──
    st.subheader("📈 Stakeholder Satisfaction")

    sat_cols = st.columns(len(resolutions))
    for col, res in zip(sat_cols, resolutions):
        with col:
            sat = res["satisfaction"]
            color = "#198754" if sat >= 80 else "#ffc107" if sat >= 50 else "#dc3545"
            st.markdown(f"""
            <div style="text-align: center; padding: 1.2rem; background: #f8f9fa;
                        border-radius: 10px; border: 3px solid {color};">
                <h1 style="color: {color}; margin: 0;">{sat}%</h1>
                <p style="color: #000; margin: 0.3rem 0 0 0; font-weight: bold;">
                    {res['icon']} {res['label']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── AI Decision Explanation ──
    if results["has_conflict"]:
        st.divider()
        st.subheader("🧠 AI Decision Explanation")

        sorted_for_explain = sorted(resolutions, key=lambda x: x["priority"])
        priority_order = " > ".join([f"{r['icon']} {r['label']}" for r in sorted_for_explain])

        st.markdown(f"""
> **Priority Order:** {priority_order}
>
> The AI allocates resources using **3GPP Release 18 priority levels** (1 = highest, 10 = lowest).
> Mission-critical services (healthcare, emergency) receive guaranteed allocation before
> entertainment and efficiency workloads.
        """)

        for res in sorted_for_explain:
            sat = res["satisfaction"]
            if sat >= 95:
                st.markdown(
                    f"- **{res['icon']} {res['label']}** (Priority {res['priority']}): "
                    f"Full {res['slice_type']} allocation with {res['bandwidth_allocated']:.0f} Mbps guaranteed."
                )
            else:
                st.markdown(
                    f"- **{res['icon']} {res['label']}** (Priority {res['priority']}): "
                    f"Received {sat}% of requested resources. *{res['adjustment']}*"
                )

        # Ethical conclusion
        critical = [r for r in sorted_for_explain if r["priority"] <= 2]
        reduced = [r for r in sorted_for_explain if r["satisfaction"] < 95]
        if critical and reduced:
            critical_names = ", ".join([f"{r['icon']} {r['label']}" for r in critical])
            reduced_names = ", ".join([f"{r['icon']} {r['label']}" for r in reduced])
            st.info(
                f"**Conclusion:** {critical_names} receive(s) priority because mission-critical "
                f"applications (human safety, emergency) take precedence over other workloads. "
                f"{reduced_names} receive(s) adjusted allocation with quality trade-offs to "
                f"maintain service within available capacity."
            )


def render_multi_intent_tab():
    """Render the Multi-Intent Conflict Resolution tab"""
    st.header("⚡ Multi-Intent Conflict Resolution")
    st.markdown("""
    **AI-Powered Negotiation** between competing network demands.
    Enter multiple stakeholder requirements and watch the AI agents negotiate
    optimal resource allocation when demands exceed network capacity.
    """)

    st.divider()

    # Quick scenario button
    st.subheader("📋 Stakeholder Requirements")

    if st.button("🔥 Load Demo: Hospital vs Stadium vs Factory", use_container_width=True):
        st.session_state.mi_intent_1 = "Emergency remote surgery at the hospital requiring ultra-low latency immediately"
        st.session_state.mi_intent_2 = "Live 4K streaming for 50000 fans at the stadium tonight with excellent quality"
        st.session_state.mi_intent_3 = "Deploy 10000 IoT sensors in the smart factory for real-time automation"
        st.rerun()

    st.markdown("")

    # Three stakeholder inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏥 Stakeholder 1**")
        intent_1 = st.text_area(
            "Requirement:",
            key="mi_intent_1",
            height=100,
            placeholder="e.g., Remote surgery at the hospital requiring ultra-low latency"
        )

    with col2:
        st.markdown("**🏟️ Stakeholder 2**")
        intent_2 = st.text_area(
            "Requirement:",
            key="mi_intent_2",
            height=100,
            placeholder="e.g., Stadium event for 50K fans with 4K streaming"
        )

    with col3:
        st.markdown("**🏭 Stakeholder 3**")
        intent_3 = st.text_area(
            "Requirement:",
            key="mi_intent_3",
            height=100,
            placeholder="e.g., IoT sensors in factory for real-time automation"
        )

    # Analyze button
    if st.button("🤖 Analyze & Resolve Conflicts", type="primary", use_container_width=True):
        intents = [i for i in [intent_1, intent_2, intent_3] if i and i.strip()]
        if len(intents) < 2:
            st.warning("Please enter at least 2 stakeholder requirements.")
            return
        run_conflict_resolution(intents)

    st.divider()

    # Show results
    render_conflict_results()


def main():
    """Main application entry point"""
    initialize_session_state()

    render_header()

    # Main content area
    tab1, tab2, tab4 = st.tabs(["🎯 Intent Processing", "📊 Network Monitor", "⚡ Multi-Intent Resolution"])

    with tab1:
        render_network_topology()
        st.divider()

        user_intent = render_intent_input()

        col1, col2 = st.columns([1, 4])
        with col1:
            execute_button = st.button("🚀 Execute", type="primary", use_container_width=True)

        if execute_button and user_intent:
            run_agents_phase1(user_intent)

        # Show reasoning form if questions are ready
        if st.session_state.reasoning_phase == 'questions_ready':
            render_reasoning_form()

        # Continue pipeline after user confirms reasoning
        if st.session_state.reasoning_phase == 'confirmed':
            run_agents_phase2()

        render_agent_activity()
        render_configuration_plan()

        # Auto-refresh for live topology
        if st.session_state.get('topology_auto_refresh', False):
            time.sleep(3)
            st.rerun()

    with tab2:
        render_metrics_dashboard()

        if st.session_state.get('monitor_auto_refresh', False):
            time.sleep(5)
            st.rerun()


    with tab4:
        render_multi_intent_tab()


if __name__ == "__main__":
    main()
