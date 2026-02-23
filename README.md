# Sec.io

An AI-powered financial analyst that answers natural language questions about public companies using SEC EDGAR filings. Ask *"What was Apple's revenue growth from FY2023 to FY2024?"* and get a structured, cited answer grounded entirely in 10-K and 10-Q data — no hallucinated numbers.

---

## How it works

A five-stage pipeline converts a raw user message into a verified, cited answer:

```mermaid
flowchart LR
    A([User Query]) --> B[Clarification\nAgent]
    B -->|needs clarification| C([Follow-up Question])
    B -->|resolved| D[Planner\nAgent]
    D --> E[Execution\nEngine]
    E -->|get_income_statement| F[(SEC EDGAR)]
    F --> E
    E -->|compute_*| G[Compute\nFunctions]
    G --> E
    E --> H[Summarizer\nAgent]
    H --> I[Hallucination\nCheck]
    I --> J([AnalysisResponse])
```

| Stage | What happens |
|---|---|
| **Clarification** | Extracts ticker, metric(s), fiscal period(s), and query type from free-form text. Asks a follow-up if confidence is below 0.85. |
| **Planning** | Produces a typed `ExecutionPlan` — an ordered list of tool calls with cross-step `$step:N:field` references. |
| **Execution** | Runs plan steps in order. Data steps call SEC EDGAR (async, TTL-cached). Compute steps call pure Python functions. |
| **Summarization** | A cheaper LLM (`gpt-4o-mini`) narrates the pre-computed results in plain English. It is explicitly prohibited from performing arithmetic. |
| **Hallucination check** | Every number in the summary is extracted and checked against the truth set of raw values and computation outputs. Unverified numbers are flagged in `guardrails.unverified_numbers`. |

---

## Core capabilities

- **Income statement retrieval** — revenue, net income, EPS, gross profit, and operating income from 10-K and 10-Q filings
- **Growth computation** — year-over-year and quarter-over-quarter growth rates with explicit formulas
- **Margin computation** — gross, operating, and net margin percentages
- **Quarterly aggregation** — sum or average quarterly values into annual figures
- **Multi-turn clarification** — conversation history is forwarded to the LLM so ambiguities can be resolved across turns
- **Scope enforcement** — queries about balance sheets, cash flows, stock prices, dividends, etc. are rejected before hitting the LLM
- **Input sanitization** — control characters stripped, length capped at 2000 characters
- **TTL cache** — EDGAR responses are cached in-process for 15 minutes to avoid redundant network calls

---

## Project structure

```
sec-llm/
├── src/sec_llm/
│   ├── main.py            # FastAPI app factory, CORS middleware, lifespan hooks
│   ├── config.py          # Settings via pydantic-settings (SEC_LLM_ prefix)
│   ├── dependencies.py    # @lru_cache DI factories for pipeline, clients, settings
│   ├── models.py          # All Pydantic schemas: errors, queries, plans, financials, responses
│   ├── compute.py         # Growth, margin, aggregation functions + COMPUTE_REGISTRY
│   ├── agents.py          # ClarificationAgentImpl, PlannerAgentImpl, SummarizerAgentImpl
│   ├── pipeline.py        # ExecutionPlanExecutor + QueryPipeline
│   ├── guardrails.py      # check_scope, sanitize_input, hallucination verification
│   ├── formatter.py       # Build citations, raw_data, computations, visualization payloads
│   ├── prompts/
│   │   ├── clarification_system.txt
│   │   ├── planner_system.txt
│   │   └── summarizer_system.txt
│   ├── api/
│   │   ├── chat.py        # POST /api/chat
│   │   ├── company.py     # GET /api/company/{ticker}
│   │   ├── health.py      # GET /api/health
│   │   └── router.py      # Aggregates all routers
│   └── sec/
│       ├── client.py      # EdgarClient — async wrapper over edgartools
│       ├── extractor.py   # Parse IncomeStatementData from filing objects
│       ├── normalizer.py  # DataFrame label matching + LABEL_CANDIDATES map
│       └── cache.py       # TTLCache (in-process, monotonic clock)
└── tests/
    ├── unit/              # Pure function tests (compute, models, formatter, guardrails)
    └── integration/       # Pipeline and API tests (mocked SEC + LLM)
```

### Data flow through models

```mermaid
classDiagram
    class UserQuery {
        +str message
        +list conversation_history
    }
    class ClarifiedQuery {
        +str ticker
        +QueryType query_type
        +list~MetricName~ metrics
        +list~FiscalPeriod~ periods
    }
    class ExecutionPlan {
        +list~PlanStep~ steps
        +str reasoning
    }
    class PlanStep {
        +int step_id
        +str tool
        +list~ToolCallArg~ args
        +list~int~ depends_on
    }
    class IncomeStatementData {
        +FilingMetadata metadata
        +float revenue
        +float net_income
        +float gross_profit
        +float operating_income
        +float eps_diluted
        +period_label() str
        +get_metric(name) float
    }
    class AnalysisResponse {
        +list raw_data
        +list computations
        +str summary
        +list~SourceCitation~ citations
        +VisualizationPayload visualization
        +GuardrailInfo guardrails
        +bool needs_clarification
    }

    UserQuery --> ClarifiedQuery : clarified by agent
    ClarifiedQuery --> ExecutionPlan : planned by agent
    ExecutionPlan "1" *-- "1..*" PlanStep
    PlanStep --> IncomeStatementData : data steps produce
    IncomeStatementData --> AnalysisResponse : serialized into raw_data
```

### Compute registry

```mermaid
flowchart TD
    R[COMPUTE_REGISTRY] --> G["compute_yoy_growth\ncompute_qoq_growth\n— both → compute_growth()"]
    R --> M["compute_margin()"]
    R --> A["aggregate_quarters()"]
    D[DATA_TOOLS] --> I["get_income_statement\n→ EdgarClient async"]
    R & D --> ALL["ALL_TOOL_NAMES\nvalidated at plan time\nand at execution time"]
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check — returns `{"status": "ok"}` |
| `GET` | `/api/company/{ticker}` | Company metadata from EDGAR (name, CIK, SIC, exchange) |
| `POST` | `/api/chat` | Main query endpoint — accepts `UserQuery`, returns `AnalysisResponse` |

### POST /api/chat

**Request**
```json
{
  "message": "What was Apple's revenue growth from FY2023 to FY2024?",
  "conversation_history": []
}
```

**Response — resolved**
```json
{
  "raw_data": [...],
  "computations": [
    {
      "metric_name": "revenue",
      "current_value": 391035000000,
      "previous_value": 383285000000,
      "growth_percentage": 2.02,
      "formula": "(391,035,000,000.00 - 383,285,000,000.00) / 383,285,000,000.00 = 2.02%"
    }
  ],
  "summary": "Apple's revenue grew 2.02% from FY2023 to FY2024...",
  "citations": [
    {"ticker": "AAPL", "filing_type": "10-K", "fiscal_period": "FY2023"},
    {"ticker": "AAPL", "filing_type": "10-K", "fiscal_period": "FY2024"}
  ],
  "visualization": {"chart_type": "comparison", "metric": "revenue", "data": [...]},
  "guardrails": {"llm_computed_math": false, "unverified_numbers": []},
  "needs_clarification": false,
  "follow_up_question": null
}
```

**Response — clarification needed**
```json
{
  "needs_clarification": true,
  "follow_up_question": "Which fiscal year and metric are you asking about?"
}
```

**Error codes**

| Status | Cause |
|---|---|
| `422` | Out-of-scope query (balance sheet, stock price, etc.) or invalid input |
| `404` | Ticker or filing not found in EDGAR |
| `429` | Rate limit exceeded (20 requests/minute per IP) |
| `502` | OpenAI API failure |

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- An OpenAI API key — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- An EDGAR identity string (your real name + email), required by the [SEC fair-use policy](https://www.sec.gov/developer)

### 1. Clone and install

```bash
git clone https://github.com/your-org/sec-llm.git
cd sec-llm
uv sync --extra dev
```

`uv sync` creates a `.venv` inside the project and installs all dependencies from `uv.lock`. The `--extra dev` flag adds pytest, ruff, and related tooling.

### 2. Configure environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set the following variables:

```bash
SEC_LLM_OPENAI_API_KEY="<OPENAI_API_KEY>"
SEC_LLM_EDGAR_IDENTITY="<Your Name your@email.com>"  # required by SEC policy
```

You can leave other environment variables as it is.

### 3. Run

```bash
uv run uvicorn sec_llm.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs (Swagger UI) at `http://localhost:8000/docs`.

### 4. Verify

```bash
# Health check
curl http://localhost:8000/api/health
# → {"status":"ok"}

# Send a query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What was Apple revenue in FY2024?"}'

# Confirm scope enforcement works
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me Apple balance sheet"}'
# → 422 with out-of-scope detail message
```

---

## Testing

```bash
# Unit + integration tests (no external calls, fast)
uv run pytest tests/unit/ tests/integration/test_api.py tests/integration/test_pipeline.py -v

# Full suite including live EDGAR calls
uv run pytest -v

# Only the live EDGAR tests
uv run pytest -m slow -v
```

---

## Supported metrics

| Key | Description | Filing source |
|---|---|---|
| `revenue` | Total net revenue / net sales | 10-K, 10-Q |
| `net_income` | Net income (loss) | 10-K, 10-Q |
| `eps` | Diluted earnings per share | 10-K, 10-Q |
| `gross_margin` | Gross profit raw value | 10-K, 10-Q |
| `operating_income` | Operating income (loss) | 10-K, 10-Q |

Only income statement data is supported. Balance sheet, cash flow, segment, and geographic data are explicitly out of scope.

---

## Limitations

- **Income statement only.** Balance sheet, cash flow, segment, and geographic breakdowns are not supported.
- **EDGAR data quality varies.** XBRL label naming is inconsistent across companies and filing years. The normalizer uses fuzzy label matching with a priority-ordered candidate list, which may miss unusual labels.
- **Fiscal year heuristics.** Filing-to-fiscal-year matching uses `period_of_report` dates and filing date ranges. Companies with non-calendar fiscal years may occasionally match the wrong filing.
- **In-process cache only.** The TTL cache is per-process and not shared across workers. For multi-worker deployments, replace `TTLCache` with a shared store (Redis, Memcached, etc.).
