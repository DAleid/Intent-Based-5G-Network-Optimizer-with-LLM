# 5G-Advanced Network Optimizer

**Agentic AI for Real-Time Optimization of 5G-Advanced Networks**

A prototype demonstrating AI-native network optimization as defined in 3GPP Release 18. This system uses multiple AI agents working together to understand user intent, configure network slices, monitor performance, and autonomously optimize the network.

## Features

### 5G-Advanced Capabilities (3GPP Release 18)
- **Intent-Based Networking**: Natural language to network configuration
- **AI-Native Optimization**: Autonomous decision-making for network optimization
- **Predictive QoS**: Proactive issue detection and prevention
- **Self-Organizing Network (SON)**: Self-configuration, self-optimization, self-healing

### Multi-Agent Architecture
1. **Intent Interpreter Agent**: Understands natural language requests
2. **Planner & Configurator Agent**: Generates optimal network configurations
3. **Monitor & Analyzer Agent**: Real-time network monitoring and analysis
4. **Optimizer & Executor Agent**: Autonomous optimization execution

### Network Slicing Support
- **eMBB**: Enhanced Mobile Broadband (high bandwidth)
- **URLLC**: Ultra-Reliable Low-Latency Communications
- **mMTC**: Massive Machine Type Communications (IoT)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone or navigate to the project directory**
```bash
cd intent-5g-optimizer
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# For Groq (FREE): Get key from https://console.groq.com
```

5. **Edit the .env file**
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application

### Start the Streamlit UI
```bash
streamlit run app.py
```

This will open the application in your web browser at `http://localhost:8501`

## Usage

### 1. Enter Your Intent
Type your network requirements in natural language:
- "Optimize for live streaming at the stadium tomorrow from 7-10 PM"
- "Emergency priority for hospital communications immediately"
- "Deploy connectivity for 10000 IoT sensors in the factory"

### 2. Watch the Agents Work
The UI shows each agent's status and output:
- Intent Interpreter analyzing your request
- Planner generating configuration
- Monitor checking network status
- Optimizer making improvements (if needed)

### 3. View Results
- **Network Dashboard**: Real-time KPIs (throughput, latency, etc.)
- **Configuration Details**: Generated slice, QoS, and RAN settings
- **Optimization Log**: History of automated optimizations

### 4. Simulate Events
Use the sidebar to simulate network events:
- Stadium Event (high capacity)
- Concert (social media heavy)
- Emergency (critical priority)
- IoT Deployment (massive connections)

## Project Structure

```
intent-5g-optimizer/
├── app.py                  # Streamlit UI (main entry)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # Environment template
│
├── agents/                # AI Agents
│   ├── intent_agent.py    # Intent Interpreter
│   ├── planner_agent.py   # Planner & Configurator
│   ├── monitor_agent.py   # Monitor & Analyzer
│   ├── optimizer_agent.py # Optimizer & Executor
│   └── crew.py           # CrewAI configuration
│
├── tools/                 # Agent Tools
│   ├── intent_tools.py    # parse_intent
│   ├── config_tools.py    # generate_config, get_templates
│   ├── monitor_tools.py   # get_metrics, check_status
│   └── action_tools.py    # execute_action, db_save, db_query
│
├── simulator/             # Network Simulator
│   └── network_sim.py     # 5G network simulation
│
├── data/                  # Templates & Data
│   └── templates.json     # 5G configuration templates
│
└── config/               # Configuration
    └── settings.py       # Application settings
```

## 5G-Advanced Standards Compliance

This prototype implements concepts from:
- **3GPP TS 38.843**: AI/ML for RAN
- **3GPP TS 28.312**: Intent-Based Management
- **3GPP TS 28.104**: Self-Optimization (SON)
- **3GPP Release 18**: 5G-Advanced specifications

## Tools Reference

| Tool | Agent | Purpose |
|------|-------|---------|
| `parse_intent` | Intent | Analyze user requests |
| `generate_config` | Planner | Create network configurations |
| `get_templates` | Planner | Access configuration templates |
| `get_metrics` | Monitor | Collect network KPIs |
| `check_status` | Monitor | Analyze and detect issues |
| `execute_action` | Optimizer | Perform optimizations |
| `db_save` | Optimizer | Save data for learning |
| `db_query` | All | Query historical data |

## LLM Providers

The system supports multiple LLM providers:

| Provider | Cost | Setup |
|----------|------|-------|
| **Groq** | FREE | `GROQ_API_KEY` from console.groq.com |
| OpenAI | Paid | `OPENAI_API_KEY` |
| Google Gemini | Free tier | `GOOGLE_API_KEY` |

## Troubleshooting

### "API key not found"
Make sure your `.env` file exists and contains the correct API key.

### "Module not found"
Run `pip install -r requirements.txt` to install all dependencies.

### Streamlit doesn't start
Try: `python -m streamlit run app.py`

## License

This is a prototype for educational and research purposes.

## Acknowledgments

- 3GPP for 5G-Advanced specifications
- CrewAI for the multi-agent framework
- Streamlit for the UI framework
