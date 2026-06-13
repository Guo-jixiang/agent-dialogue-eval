# PromptScope - Dialogue Instruction Evaluation Platform

PromptScope is an automated evaluation platform for conversational AI products. It simulates real user interactions with your dialogue model across diverse personas, then scores the model on 7 dimensions using a panel of LLM judges, producing quantifiable, traceable, and comparable evaluation reports.

## Features

- **Automatic Dialogue Simulation** - 4-dimension persona system (cooperation, verbosity, familiarity, urgency) generating up to 72 unique user behavior combinations
- **7-Dimension Scoring** - Flow adherence, constraint compliance, knowledge accuracy, style/tone, coherence, safety, and adaptability
- **Self-Consistency Checking** - Multiple independent evaluations per checkpoint to reduce scoring variance
- **Anomaly Detection** - Automatic flagging of suspicious scores, bimodal distributions, and low-confidence results
- **Evidence Traceability** - Every score is backed by specific dialogue turn references and quoted text
- **3-Layer Analysis Chain** - Per-dialogue analysis (ReportAgent) -> cross-comparison (JudgeAgent) -> root cause attribution (AttributionAgent)
- **Checkpoint/Resume** - Evaluation runs survive interruptions and resume from the last completed batch
- **Multiple Interfaces** - CLI, Web UI (Vue.js), and REST API with SSE streaming
- **Prompt Optimization** - AI-generated suggestions to improve your task instructions based on evaluation findings
- **Built-in AI Assistant** - Chat with an assistant that understands your evaluation reports
- **Multi-format Reports** - HTML (with charts), Markdown, and JSON output

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Instruction (System Prompt)                              │
└────────────────────┬─────────────────────────────────────┘
                     │ Phase 1: Parse
                     ▼
┌──────────────────────────────────────────────────────────┐
│  InstructionSpec (flow graph, constraints, knowledge)     │
└────────────────────┬─────────────────────────────────────┘
                     │ Phase 2: Generate Personas
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Persona Matrix (4 dimensions × N combinations)           │
└────────────────────┬─────────────────────────────────────┘
                     │ Phase 3: Simulate + Evaluate
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Dialogue 1│ │Dialogue 2│ │Dialogue N│   (parallel)
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ 7 Judges│ │ 7 Judges│ │ 7 Judges│   (parallel per dialogue)
    └────┬────┘ └────┬────┘ └────┬────┘
         └───────────┼───────────┘
                     │ Phase 4: Analysis
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   Layer 4      Layer 5      Layer 6
   ReportAgent  JudgeAgent   AttributionAgent
   (per-dialog) (cross-comp) (root cause)
                     │
                     ▼
              ┌─────────────┐
              │ Final Report │
              │ HTML/MD/JSON │
              └─────────────┘
```

## Scoring Dimensions

| Dimension | Key | Default Weight | Description |
|-----------|-----|---------------|-------------|
| Flow Adherence | `flow` | 25% | Whether the agent follows the prescribed dialogue flow |
| Constraint Compliance | `constraint` | 20% | Adherence to hard/soft constraints (word limits, forbidden terms, etc.) |
| Knowledge Accuracy | `knowledge` | 20% | Correctness of business information against knowledge points |
| Style & Tone | `style` | 10% | Whether expression style matches the defined role |
| Coherence | `coherence` | 10% | Contextual logic consistency, no contradictions |
| Safety | `safety` | 10% | No harmful content, no system prompt leakage, no hallucination |
| Adaptability | `adaptability` | 5% | Handling of unexpected user behaviors and error recovery |

**Hard constraint penalty:** Any hard constraint violation applies a tiered penalty (1 failure: x0.9, 2: x0.8, 3: x0.7, 4+: x0.6).

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the web frontend)
- An OpenAI-compatible LLM API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
# Judge LLM (recommend GPT-4o or equivalent)
EVAL_LLM_API_KEY=your_key
EVAL_LLM_BASE_URL=https://api.openai.com/v1
EVAL_LLM_MODEL=gpt-4o

# User simulation LLM (lower cost model is fine)
SIM_LLM_API_KEY=your_key
SIM_LLM_BASE_URL=https://api.openai.com/v1
SIM_LLM_MODEL=gpt-4o-mini

# Agent under test (the model being evaluated)
AGENT_LLM_API_KEY=your_key
AGENT_LLM_BASE_URL=https://api.openai.com/v1
AGENT_LLM_MODEL=your_model_name
```

### Option 1: Web Console (Recommended)

```bash
# Build frontend (first time only)
cd web && npm install && npm run build && cd ..

# Start server
python cli.py serve --port 8000
```

Open `http://localhost:8000` in your browser.

For development with hot-reload:
```bash
python cli.py serve --port 8000 --dev
```

### Option 2: CLI

```bash
# Full evaluation (all personas)
python cli.py run --instruction my_task.txt

# Specific personas only
python cli.py run --instruction my_task.txt --personas cooperative,resistant,impatient

# Parse instruction structure (no evaluation)
python cli.py parse --instruction my_task.txt

# Simulate a single dialogue (no scoring, for debugging)
python cli.py simulate --instruction my_task.txt --persona resistant --max-turns 20
```

### Option 3: REST API

```bash
# Run evaluation (SSE streaming)
curl -N -X POST http://localhost:8000/api/eval/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "You are a customer service agent...", "agent_type": "llm"}'

# Evaluate an existing dialogue (no simulation)
curl -X POST http://localhost:8000/api/eval/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "You are...",
    "dialogue": [
      {"role": "agent", "content": "Hello, how can I help?"},
      {"role": "user", "content": "I need help with my order."}
    ]
  }'

# Get report
curl http://localhost:8000/api/eval/report/{report_id}?fmt=html

# List recent reports
curl http://localhost:8000/api/eval/reports
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/eval/parse` | Parse instruction into structured spec |
| `POST` | `/api/eval/run` | Run full evaluation (SSE streaming) |
| `POST` | `/api/eval/evaluate` | Evaluate an externally-provided dialogue |
| `POST` | `/api/eval/simulate` | Simulate a single dialogue |
| `POST` | `/api/eval/generate-personas` | AI-generate test personas |
| `POST` | `/api/eval/optimize-prompt` | Generate instruction optimization suggestions |
| `GET` | `/api/eval/reports` | List recent reports |
| `GET` | `/api/eval/report/{id}` | Get report (`?fmt=json/html/md`) |
| `DELETE` | `/api/eval/report/{id}` | Delete a report |
| `POST` | `/api/assistant/chat` | Built-in AI assistant (SSE streaming) |

## Project Structure

```
.
├── api/                    # FastAPI web server
│   ├── app.py              # App factory
│   └── routes.py           # All API endpoints
├── cli.py                  # CLI entry point (Click)
├── config/
│   ├── prompts/            # Jinja2 prompt templates (17 templates)
│   └── settings.py         # Pydantic settings (from .env)
├── core/
│   ├── agent.py            # Agent abstraction (LLM + Friday)
│   ├── llm_client.py       # OpenAI-compatible LLM client
│   ├── models.py           # All Pydantic data models
│   └── rate_limit.py       # Token bucket rate limiter
├── evaluator/
│   ├── judges/             # 7 dimension judges + self-consistency wrapper
│   ├── anomaly.py          # Anomaly detection
│   ├── attribution_agent.py # Layer 6: root cause analysis
│   ├── decomposer.py       # Instruction → checkpoints
│   ├── eval_pipeline.py    # Single dialogue evaluation pipeline
│   ├── evidence.py         # Evidence summarization
│   ├── judge_agent.py      # Layer 5: cross-comparison
│   ├── report_agent.py     # Layer 4: per-dialogue analysis
│   └── scorer.py           # Score aggregation
├── parser/
│   └── instruction_parser.py # LLM-based instruction parsing
├── report/
│   ├── generator.py        # Multi-format report generation
│   └── templates/          # HTML/Markdown report templates
├── simulator/
│   ├── engine.py           # Legacy dialogue engine wrapper
│   ├── persona.py          # Persona generation & enrichment
│   ├── persona_generator.py # AI-powered persona generation
│   ├── runner.py           # Conversation runner
│   ├── scenario.py         # Scenario matrix execution
│   └── user_simulator.py   # User behavior simulation
├── tests/                  # pytest test suite
├── web/                    # Vue.js frontend
├── Dockerfile              # Container build
├── docker-compose.yml      # Docker Compose config
├── deploy.sh               # One-click deploy to cloud server
└── run_eval.py             # Main orchestration pipeline
```

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_LLM_*` | - | Judge panel LLM (API key, base URL, model) |
| `SIM_LLM_*` | - | User simulation LLM |
| `AGENT_LLM_*` | - | Agent under test |
| `ASSISTANT_LLM_*` | Falls back to EVAL | Built-in AI assistant LLM |
| `MAX_TURNS` | 30 | Maximum dialogue turns |
| `SIM_CONCURRENCY` | 8 | Concurrent dialogue simulations |
| `EVAL_CONCURRENCY` | 8 | Concurrent evaluations |
| `EVAL_BATCH_SIZE` | 12 | Personas per batch (checkpoint granularity) |
| `SELF_CONSISTENCY_RUNS` | 3 | Repeated evaluations per subjective checkpoint |
| `LLM_RATE_LIMIT_RPS` | 0 (disabled) | Requests per second cap |
| `WEIGHT_FLOW` | 0.25 | Flow dimension weight |
| `WEIGHT_CONSTRAINT` | 0.20 | Constraint dimension weight |
| `WEIGHT_KNOWLEDGE` | 0.20 | Knowledge dimension weight |
| `WEIGHT_STYLE` | 0.10 | Style dimension weight |
| `WEIGHT_COHERENCE` | 0.10 | Coherence dimension weight |
| `WEIGHT_SAFETY` | 0.10 | Safety dimension weight |
| `WEIGHT_ADAPTABILITY` | 0.05 | Adaptability dimension weight |

Dimension weights are auto-normalized, so they only need to maintain correct relative proportions.

## Deployment

### Docker

```bash
docker compose up -d
```

### Cloud Server (One-Click)

Edit the configuration section at the top of `deploy.sh`, then:

```bash
chmod +x deploy.sh
./deploy.sh
```

## License

See the project for license details.
