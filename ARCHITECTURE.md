# Eidetic — Technical Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         WEB UI (React)                            │
│  [New Project] → Research Question → [Start]                      │
│  Dashboard: knowledge graph + propositions + convergence + diffs  │
│  SSE stream ◄──────────────────────────────────────┐              │
└──────────────────────┬─────────────────────────────┼──────────────┘
                       │                             │
                       ▼                             │
              ┌─────────────────┐                    │
              │   BACKEND API   │ FastAPI (Python)    │
              │   (Koyeb)       │────────────────────►│
              └──┬──────────┬───┘   SSE events        │
                 │          │                          │
    ┌────────────┘          └──────────────┐          │
    ▼                                      ▼          │
┌─────────────┐                  ┌──────────────────┐ │
│ ElevenLabs  │  post-call       │  Mistral API     │ │
│ Agent       │  webhook ──────► │  (Large)         │ │
│ (voice)     │                  │  Designer/       │ │
│             │ ◄── PATCH prompt │  Analyst/        │ │
│ Custom LLM: │                  │  Synthesizer     │ │
│ Mistral Lg  │                  └──────────────────┘ │
└─────────────┘                                       │
                                                      │
              ┌───────────────────────────────────────┘
              │
   ┌──────────▼──────────┐
   │   /data/projects/   │  JSON file storage
   │   {project_id}/     │
   │     evidence.json   │
   │     propositions.json│
   │     interviews/     │
   │     scripts/        │
   └─────────────────────┘
```

## Directory Structure

```
eidetic/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                    # LLM config, model selection, feature flags
├── main.py                      # FastAPI app entry point
│
├── agents/
│   ├── __init__.py
│   ├── llm_client.py           # Unified LLM wrapper (provider-agnostic)
│   ├── designer.py             # Interview script generation
│   ├── analyst.py              # Evidence extraction + proposition management
│   └── synthesizer.py          # Final report generation
│
├── models/
│   ├── __init__.py
│   ├── evidence.py             # Evidence dataclass/schema
│   ├── proposition.py          # Proposition dataclass/schema
│   ├── interview.py            # Interview transcript schema
│   ├── script.py               # Interview script schema
│   └── project.py              # Project container
│
├── services/
│   ├── __init__.py
│   ├── elevenlabs_service.py   # ElevenLabs API (create agent, PATCH prompt, etc.)
│   ├── project_service.py      # Project CRUD, data persistence
│   ├── pipeline.py             # Orchestrates: webhook → analyst → designer → PATCH
│   └── sse_manager.py          # Server-Sent Events for dashboard
│
├── api/
│   ├── __init__.py
│   ├── routes_projects.py      # /api/projects/* endpoints
│   ├── routes_webhook.py       # /api/webhook/elevenlabs
│   └── routes_stream.py        # /api/projects/{id}/stream (SSE)
│
├── prompts/
│   ├── designer_system.txt     # Designer agent system prompt
│   ├── analyst_system.txt      # Analyst agent system prompt
│   ├── synthesizer_system.txt  # Synthesizer agent system prompt
│   └── interviewer_base.txt    # Base interviewer prompt (before script injection)
│
├── dashboard/                   # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── KnowledgeGraph.jsx    # d3.js force-directed graph
│   │   │   ├── PropositionList.jsx   # Sortable proposition table
│   │   │   ├── ConvergenceMeter.jsx  # Visual convergence indicator
│   │   │   ├── ScriptDiff.jsx        # Script version comparison
│   │   │   ├── InterviewTimeline.jsx # Interview list with key findings
│   │   │   └── ProjectSetup.jsx      # New project form
│   │   └── hooks/
│   │       └── useSSE.js             # SSE connection hook
│   └── public/
│
├── training/                    # Fine-tuning pipeline (optional)
│   ├── transcribe.py           # Voxtral transcription
│   ├── extract_examples.py     # Training data extraction
│   ├── anonymize.py            # PII + Yandex reference removal
│   ├── format_jsonl.py         # JSONL formatting for Mistral API
│   └── finetune.py             # Launch fine-tuning job
│
└── data/
    └── projects/               # Runtime data (gitignored)
        └── .gitkeep
```

## Component Details

### 1. LLM Client (`agents/llm_client.py`)

Provider-agnostic wrapper. All agents call this instead of Mistral directly.

```python
import httpx

class LLMClient:
    def __init__(self, config: dict):
        self.provider = config["provider"]
        self.model = config["model"]
        self.api_base = config.get("api_base", self._default_base())
        self.api_key = config["api_key"]

    def _default_base(self):
        bases = {
            "mistral": "https://api.mistral.ai/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }
        return bases[self.provider]

    async def chat(self, messages: list, temperature: float = 0.7,
                   max_tokens: int = 4096, response_format: dict = None) -> str:
        """Send chat completion request. Returns text response."""
        # All providers use OpenAI-compatible format
        # (Anthropic needs adapter, but for hackathon = Mistral only)
        ...

    async def chat_json(self, messages: list, **kwargs) -> dict:
        """Chat with JSON response parsing."""
        ...
```

**Key design**: Every prompt is in `prompts/` as plain text. No provider-specific formatting. The wrapper handles auth headers, response parsing, error handling.

### 2. Designer Agent (`agents/designer.py`)

```python
class DesignerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("designer_system.txt")

    async def generate_initial_script(self, research_question: str,
                                       initial_angles: list[str] = None) -> tuple[list[Proposition], InterviewScript]:
        """Generate initial propositions and first interview script."""
        ...

    async def update_script(self, research_question: str,
                            propositions: list[Proposition],
                            evidence: list[Evidence],
                            previous_script: InterviewScript) -> InterviewScript:
        """Generate updated script based on current knowledge state."""
        ...

    def _build_interviewer_prompt(self, script: InterviewScript) -> str:
        """Convert InterviewScript to full system prompt for ElevenLabs agent."""
        ...
```

### 3. Analyst Agent (`agents/analyst.py`)

The core intelligence. Runs after each interview.

```python
class AnalystAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("analyst_system.txt")

    async def analyze_interview(self, transcript: str,
                                 existing_evidence: list[Evidence],
                                 existing_propositions: list[Proposition]) -> AnalysisResult:
        """
        Full analysis pipeline in one LLM call:
        1. Extract evidence (Factor → Mechanism → Outcome)
        2. Map each evidence to existing propositions (supports/contradicts/orphan)
        3. Generate new propositions from orphan evidence
        4. Retroactive scan: check all existing evidence against new propositions
        5. Recalculate confidence scores
        6. Identify merge/prune candidates
        7. Determine system mode (divergent/convergent)

        Returns: AnalysisResult with all updates
        """
        ...

    async def retroactive_scan(self, new_propositions: list[Proposition],
                                all_evidence: list[Evidence]) -> list[EvidenceMapping]:
        """Check all existing evidence against newly generated propositions."""
        ...
```

### 4. Pipeline Orchestrator (`services/pipeline.py`)

Coordinates the flow triggered by webhooks.

```python
class Pipeline:
    def __init__(self, analyst: AnalystAgent, designer: DesignerAgent,
                 elevenlabs: ElevenLabsService, sse: SSEManager):
        self.analyst = analyst
        self.designer = designer
        self.elevenlabs = elevenlabs
        self.sse = sse

    async def process_interview(self, project_id: str, transcript: str,
                                 conversation_id: str):
        """
        Full pipeline triggered by post-call webhook:
        1. Save transcript to interview store
        2. Run analyst → get new evidence + updated propositions
        3. Emit SSE events for each new evidence and proposition update
        4. Run designer → generate updated script
        5. PATCH ElevenLabs agent with new system prompt
        6. Save new script version
        7. Emit SSE event for script update
        """
        # Step 1: Save
        project = load_project(project_id)
        interview = save_interview(project, transcript, conversation_id)

        # Step 2: Analyze
        result = await self.analyst.analyze_interview(
            transcript=transcript,
            existing_evidence=project.evidence_store,
            existing_propositions=project.proposition_store
        )

        # Step 3: Emit events
        for evidence in result.new_evidence:
            project.evidence_store.append(evidence)
            await self.sse.emit(project_id, "new_evidence", evidence.dict())

        for prop_update in result.proposition_updates:
            update_proposition(project, prop_update)
            await self.sse.emit(project_id, "proposition_updated", prop_update.dict())

        for new_prop in result.new_propositions:
            project.proposition_store.append(new_prop)
            await self.sse.emit(project_id, "new_proposition", new_prop.dict())

        for merge in result.merges:
            apply_merge(project, merge)
            await self.sse.emit(project_id, "proposition_merged", merge.dict())

        # Step 4: New script
        new_script = await self.designer.update_script(
            research_question=project.research_question,
            propositions=project.proposition_store,
            evidence=project.evidence_store,
            previous_script=project.current_script
        )
        project.script_versions.append(new_script)

        # Step 5: PATCH ElevenLabs
        full_prompt = self.designer._build_interviewer_prompt(new_script)
        await self.elevenlabs.update_agent_prompt(project.elevenlabs_agent_id, full_prompt)

        # Step 6-7: Save and emit
        save_project(project)
        await self.sse.emit(project_id, "script_updated", {
            "version": new_script.version,
            "changes_summary": new_script.changes_summary
        })
```

### 5. ElevenLabs Service (`services/elevenlabs_service.py`)

```python
class ElevenLabsService:
    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"xi-api-key": api_key}

    async def create_agent(self, name: str, system_prompt: str,
                           first_message: str, mistral_api_key: str,
                           webhook_url: str) -> str:
        """Create ElevenLabs agent with Mistral Custom LLM. Returns agent_id."""
        payload = {
            "name": name,
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": system_prompt
                    },
                    "first_message": first_message,
                    "language": "en"
                },
                "tts": {
                    "voice_id": "<selected_voice_id>"
                }
            },
            # Custom LLM and webhook configured via dashboard
            # (easier than API for initial setup)
        }
        # POST /v1/convai/agents
        ...
        return agent_id

    async def update_agent_prompt(self, agent_id: str, new_prompt: str):
        """PATCH agent system prompt."""
        payload = {
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": new_prompt
                    }
                }
            }
        }
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{self.BASE_URL}/convai/agents/{agent_id}",
                json=payload,
                headers=self.headers
            )
            resp.raise_for_status()

    async def get_conversation_link(self, agent_id: str) -> str:
        """Get public talk-to link for the agent."""
        return f"https://elevenlabs.io/app/talk-to/{agent_id}"
```

### 6. SSE Manager (`services/sse_manager.py`)

Pushes real-time updates to the dashboard.

```python
import asyncio
from collections import defaultdict

class SSEManager:
    def __init__(self):
        self.subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, project_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.subscribers[project_id].append(queue)
        return queue

    def unsubscribe(self, project_id: str, queue: asyncio.Queue):
        self.subscribers[project_id].remove(queue)

    async def emit(self, project_id: str, event_type: str, data: dict):
        for queue in self.subscribers[project_id]:
            await queue.put({"event": event_type, "data": data})
```

---

## ElevenLabs Agent Setup (Manual Steps)

These steps are done once via the ElevenLabs dashboard before the code pipeline runs:

1. **Create Agent** → Blank template
2. **LLM** → Custom LLM:
   - Server URL: `https://api.mistral.ai/v1`
   - Model ID: `mistral-large-latest`
   - Secret: Create secret with Mistral API key
3. **Voice** → Pick a neutral, friendly voice from library
4. **Language** → English (primary), add: French, Japanese, Russian
5. **System Tools** → Enable `end_call` and `language_detection`
6. **Security** → Enable "System prompt" override (for PATCH API)
7. **Post-call webhook** → Set URL to `https://{koyeb-app-url}/api/webhook/elevenlabs`
   - Enable transcription webhook
   - Enable audio webhook (for recordings)
8. **System Prompt** → Paste initial prompt (will be overwritten by PATCH)
9. **First Message** → "Hey! I'm Sasha, an AI research assistant. I'm studying what people think about this hackathon. Could you share any thoughts — literally anything that comes to mind?"
10. **Save** → Note the `agent_id`

---

## Configuration

### Environment Variables (`.env`)

```bash
# Mistral
MISTRAL_API_KEY=your_mistral_api_key

# ElevenLabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_AGENT_ID=your_agent_id

# App
APP_BASE_URL=https://your-app.koyeb.app
DATA_DIR=./data/projects
DEFAULT_PROJECT_ID=hackathon-demo

# Optional: Fine-tuning
WANDB_API_KEY=your_wandb_key
```

### Model Configuration (`config.py`)

```python
import os

LLM_CONFIG = {
    "designer": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "analyst": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "temperature": 0.3,  # Lower for more consistent extraction
        "max_tokens": 8192,
    },
    "synthesizer": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "temperature": 0.5,
        "max_tokens": 8192,
    },
    "interviewer": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "api_key": os.getenv("MISTRAL_API_KEY"),
        # This config is for reference; actual LLM runs through ElevenLabs
    },
}

# Convergence thresholds
CONVERGENCE_SCORE_THRESHOLD = 0.6
NOVELTY_RATE_THRESHOLD = 0.15
MERGE_OVERLAP_THRESHOLD = 0.6
PRUNE_CONFIDENCE_THRESHOLD = 0.15
PRUNE_MIN_INTERVIEWS = 3

# Interview constraints
MAX_INTERVIEW_DURATION_MINUTES = 10
MAX_PROPOSITIONS_IN_SCRIPT = 8
```

---

## Deployment (Koyeb)

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Koyeb Setup

1. Connect GitHub repo: `shipaleks/mistral-hackaton`
2. Build: Docker
3. Environment variables: set all from `.env`
4. Region: Frankfurt (closest to Paris)
5. Instance: Nano ($5.4/mo) — sufficient for hackathon

### Frontend

For hackathon: serve React build from FastAPI static files. No separate deployment needed.

```python
# main.py
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="dashboard/build", html=True), name="frontend")
```

---

## Testing Strategy

### Local Testing (Before Live Demo)

1. Create test project: `POST /api/projects` with test research question
2. Simulate 3-5 interviews by calling the analyst directly with fake transcripts:
   ```
   POST /api/projects/test-01/simulate
   {"transcript": "Agent: Tell me about... User: Well, I think..."}
   ```
3. Verify: evidence extracted correctly, propositions generated, script updated
4. Delete test project: `DELETE /api/projects/test-01`

### Voice Testing

1. Open ElevenLabs agent link
2. Conduct 3-5 interviews yourself (different personas: happy, critical, tangential)
3. Check dashboard updates after each
4. Verify script evolution
5. Delete test project, create fresh `hackathon-demo` for live use

### Fallback Plan

If ElevenLabs has issues → text-mode interviews via web UI (type instead of speak). Same pipeline, same analysis, just no voice. Dashboard still works.

If Mistral API is slow → already on Large everywhere, can't go bigger. Accept latency. For interviewer: switch to `mistral-medium-latest` if unbearable.

If webhook fails → manual transcript paste into admin UI. Pipeline processes it the same way.
