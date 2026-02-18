# Intent-Based 5G Network Optimizer with LLM

**Agentic AI for Autonomous Optimization of 5G-Advanced Radio Access Networks**

A working prototype of an intent-driven network management system built on 3GPP Release 18 principles. Natural language intents are parsed by a Groq-powered LLM agent pipeline and translated into network configurations, while the system continuously monitors KPIs and heals itself autonomously.

---

## Overview

Traditional network management requires operators to manually translate business goals into low-level RAN parameters. This system eliminates that gap:

- An operator types: *"Prioritize emergency communications at the hospital now"*
- Six AI agents process the request end-to-end in under 30 seconds
- The network is reconfigured, monitored, and self-healed — no manual steps

The key architectural principle is **LLM translates, rules validate, policies execute**. The LLM is never given direct control over the network; it only parses intent and proposes configurations. A deterministic rule-based validator acts as the safety gate before any action is taken.

---

## Architecture

```
User (Natural Language Intent)
        │
        ▼
┌───────────────────────────────────────────────┐
│              Streamlit UI (app.py)             │
│   Tab 1: Intent Processing                     │
│   Tab 2: Network Monitor + Topology            │
│   Tab 3: Multi-Intent Conflict Resolution      │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│         CrewAI Agent Pipeline (Groq LLM)       │
│                                               │
│  Agent 1 → Intent Parser      (LLM)           │
│  Agent 2 → Reasoner           (LLM)           │
│  Agent 3 → Validator          (Rule-based)    │  ← Safety gate
│  Agent 4 → Planner            (LLM)           │
│  Agent 5 → Monitor            (LLM)           │
│  Agent 6 → Optimizer          (LLM)           │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│    Network Simulator + Real Dataset            │
│    6G HetNet CSV · 5,000 rows · 49 Cell_IDs   │
└───────────────────────────────────────────────┘
```

---

## Agent Pipeline

| # | Agent | Powered By | Role |
|---|-------|-----------|------|
| 1 | **Intent Parser** | Groq LLM | Converts natural language into a structured intent object (type, priority, slice, timing, SLA targets) |
| 2 | **Reasoner** | Groq LLM | Asks clarifying questions, assesses feasibility, identifies risks, simulates impact |
| 3 | **Validator** | Rule-based | Hard safety gate — enforces bandwidth caps, user limits, confidence thresholds. Cannot be overridden by LLM |
| 4 | **Planner** | Groq LLM | Generates a 3GPP-compliant network configuration (slice type, QoS, RAN params) |
| 5 | **Monitor** | Groq LLM | Reads real KPIs from the dataset/simulator, detects anomalies, triggers alerts |
| 6 | **Optimizer** | Groq LLM | Executes corrective actions, with automatic rollback if metrics degrade |

### Why Agent 3 is Rule-Based

The Validator enforces hard limits that must be deterministic and auditable:

| Limit | Value | Reason |
|-------|-------|--------|
| Max bandwidth | 500 Mbps | Physical capacity |
| Min bandwidth | 10 Mbps | Minimum viable allocation |
| Min confidence | 70% | Reject uncertain LLM output |
| Max cells active at once | 5 | Prevent mass activation |
| Min active cells | 5 | Maintain coverage |

LLMs are non-deterministic. Safety constraints must not be.

---

## Dataset

**File:** `data/6G_HetNet_Transmission_Management.csv`

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 24 |
| Unique Cell_IDs | 49 (range 1–49) |
| Cell types | Macro (10), Micro (17), Pico (8), Femto (14) |

**Key columns used:**

| Column | Used For |
|--------|----------|
| `Cell_ID` | Pinning topology cells to specific real cells |
| `Cell_Type` | Macro / Micro / Pico / Femto classification |
| `Achieved_Throughput_Mbps` | Live KPI display |
| `Network_Latency_ms` | SLA monitoring |
| `Resource_Utilization` | Cell load % |
| `Signal_to_Noise_Ratio_dB` | RAN quality metric |
| `Interference_Level_dB` | Interference monitoring |
| `QoS_Satisfaction` | Service quality score |
| `Packet_Loss_Ratio` | Reliability metric |

**How the dataset is used:**

- **Simulator:** Reads one row per `get_metrics()` call, cycling through all 5,000 records sequentially
- **Topology:** Each of the 12 HetNet topology cells is pinned to a dedicated `Cell_ID` so every cell displays real, distinct KPI data from its own real-world counterpart

---

## Network Topology

The UI renders a 12-cell Heterogeneous Network (HetNet) with 4 cell tiers:

| Cell | Type | Dataset Cell_ID | Real rows |
|------|------|----------------|-----------|
| C01 Central | Macro | 3 | 103 |
| C02 North | Macro | 4 | 108 |
| C03 South | Macro | 13 | 98 |
| C04 Micro-1 | Micro | 2 | 99 |
| C05 Micro-2 | Micro | 7 | 78 |
| C06 Micro-3 | Micro | 12 | 99 |
| C07 Micro-4 | Micro | 14 | 94 |
| C08 Pico-1 | Pico | 9 | 98 |
| C09 Pico-2 | Pico | 11 | 102 |
| C10 Pico-3 | Pico | 18 | 95 |
| C11 Femto-1 | Femto | 1 | 102 |
| C12 Femto-2 | Femto | 5 | 101 |

Cell health (healthy / warning / critical) is computed from real throughput, latency, and load values.

---

## Supported Intent Types

| Intent | Example Phrase | Slice |
|--------|---------------|-------|
| Stadium Event | "Optimize for the match tonight" | eMBB |
| Emergency | "Emergency priority at the hospital" | URLLC |
| IoT Deployment | "10,000 sensors in the factory" | mMTC |
| Healthcare | "Low latency for telemedicine" | URLLC |
| Video Streaming | "High quality live stream" | eMBB |
| Smart Factory | "Industrial automation connectivity" | URLLC |
| Gaming | "Minimize latency for gaming" | URLLC |
| Concert / Event | "Social media heavy event" | eMBB |
| Transportation | "Vehicle connectivity on the highway" | URLLC |
| General Optimization | "Improve network performance" | eMBB |

---

## Multi-Intent Conflict Resolution

When multiple stakeholders submit competing intents simultaneously, the system resolves conflicts by:

1. Ranking intents by priority (Emergency > Healthcare > Stadium > General)
2. Allocating bandwidth from a shared 500 Mbps pool
3. Identifying resource conflicts (overlapping cells, spectrum)
4. Producing a negotiated configuration that satisfies as many SLAs as possible

---

## KPI Thresholds

| KPI | Target | Warning | Critical |
|-----|--------|---------|----------|
| Latency | < 50 ms | < 80 ms | ≥ 100 ms |
| Throughput | > 100 Mbps | > 60 Mbps | ≤ 30 Mbps |
| Packet Loss | < 0.01% | < 0.1% | ≥ 1.0% |
| Cell Load | < 70% | < 85% | ≥ 95% |

---

## 3GPP Standards Compliance

| Standard | Relevance |
|----------|-----------|
| 3GPP TS 28.312 | Intent-Based Management |
| 3GPP TS 38.843 | AI/ML for RAN |
| 3GPP TS 28.104 | Self-Optimization (SON) |
| 3GPP Release 18 | 5G-Advanced specifications |
| TM Forum IG1230 | Autonomous Networks (Level 5) |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Agent Framework | CrewAI 0.86.0 |
| LLM | Groq — Llama 3.3 70B Versatile |
| LLM Router | LiteLLM (via CrewAI) |
| Visualization | Plotly |
| Data | Pandas / NumPy |
| Language | Python 3.10+ |

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com)

### Install

```bash
git clone https://github.com/DAleid/Intent-Based-5G-Network-Optimizer-with-LLM.git
cd Intent-Based-5G-Network-Optimizer-with-LLM

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Streamlit Cloud Deployment

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** and select this repository
4. Set **Main file path** to `app.py`
5. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

6. Click **Deploy**

The dataset is included in the repository so no external data source is needed.

---

## Project Structure

```
intent-5g-optimizer-main/
├── app.py                        # Streamlit UI — main entry point
├── requirements.txt
├── .env.example                  # Environment variable template
│
├── agents/
│   ├── crew.py                   # CrewAI crew definition + Groq LLM config
│   ├── llm_client.py             # LangChain-Groq client for tool-level LLM calls
│   ├── intent_agent.py           # Intent Parser agent definition
│   ├── planner_agent.py          # Planner agent definition
│   ├── monitor_agent.py          # Monitor agent definition
│   └── optimizer_agent.py        # Optimizer agent definition
│
├── tools/
│   ├── intent_tools.py           # parse_intent (LLM + keyword fallback)
│   ├── config_tools.py           # generate_config, get_templates
│   ├── monitor_tools.py          # get_metrics, check_status
│   ├── action_tools.py           # execute_action, rollback
│   └── reasoning_llm.py          # Feasibility, risk, impact reasoning
│
├── simulator/
│   └── network_sim.py            # Dataset-driven 5G network simulator
│
├── config/
│   └── settings.py               # LLM config, KPI thresholds, network limits
│
└── data/
    ├── 6G_HetNet_Transmission_Management.csv   # Real dataset (5,000 rows)
    └── templates.json            # 3GPP slice configuration templates
```

---

## Troubleshooting

**`GROQ_API_KEY` not found**
Make sure `.env` exists with your key, or add it in Streamlit Cloud secrets.

**Rate limit error (429)**
The Groq free tier allows 12,000 tokens/minute. The app has automatic retry with exponential backoff (5s → 10s → 20s), but running multiple pipelines back-to-back may still trigger it. Wait 60 seconds and try again.

**Module not found**
Run `pip install -r requirements.txt` inside your virtual environment.

**Streamlit won't start**
Try `python -m streamlit run app.py`.

---

## License

This project is released for educational and research purposes.

## References

1. 3GPP TS 28.312 — Intent driven management services for mobile networks
2. 3GPP Release 18 — 5G-Advanced specifications
3. TM Forum IG1230 — Autonomous Networks Technical Architecture
4. ETSI ZSM — Zero-touch network and Service Management
