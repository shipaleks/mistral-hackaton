# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eidetic MVP — an autonomous AI-powered qualitative research system. It conducts adaptive voice interviews via ElevenLabs, extracts causal propositions from transcripts using Mistral AI, and autonomously evolves its interview strategy based on accumulating evidence. Implements automated grounded theory methodology.

Built for the Mistral AI Worldwide Hackathon 2026.

## Commands

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys

# Run dev server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest -q

# Run a single test file
pytest tests/test_pipeline.py -q

# Run a specific test
pytest tests/test_pipeline.py::test_name -q

# Simulate an interview locally (no ElevenLabs needed)
curl -X POST http://localhost:8000/api/projects/demo/simulate \
  -H "Content-Type: application/json" \
  -d '{"transcript":"User: Time pressure helped us focus"}'

# Batch import transcripts (manual recovery if webhook missed calls)
python scripts/import_transcripts.py ./transcripts \
  --app-url http://localhost:8000 --project-id <project_id>

# Inspect project state
python scripts/show_project.py <project_id>
```

API docs: `http://localhost:8000/docs` (Swagger UI)
Demo UI: `http://localhost:8000/ui`

## Architecture

### Multi-Agent Pipeline

Four LLM agents orchestrated through `services/pipeline.py`:

1. **Designer** (`agents/designer.py`) — generates/updates interview scripts based on current proposition state
2. **Interviewer** (ElevenLabs external agent) — conducts voice interviews using the script
3. **Analyst** (`agents/analyst.py`) — extracts evidence from transcripts, manages propositions (create/merge/prune)
4. **Synthesizer** (`agents/synthesizer.py`) — generates final research reports grounded in evidence

All agents call through `agents/llm_client.py`, a raw HTTP wrapper over the Mistral chat completions API (uses `httpx`, not the Mistral SDK). Supports `chat()` for text and `chat_json()` for structured JSON responses with `response_format={"type": "json_object"}`.

### Core Data Flow

```
ElevenLabs webhook (POST /api/webhook/elevenlabs)
  → Validate signature, extract transcript
  → Duplicate check (conversation_id)
  → Return 200 OK immediately
  → BackgroundTasks: pipeline.process_interview()
    → Save interview to disk (guaranteed)
    → Analyst extracts evidence & updates propositions
      └ on failure: save partial state + raise
    → Designer generates new script (with fallback)
      └ on failure: save analysis results + raise
    → ScriptSafetyGuard validates/sanitizes script
    → ElevenLabs agent prompt patched via API
      └ on failure: mark sync_pending, continue
    → SSE events emitted to dashboard
    → Next interview uses evolved script
```

The webhook is **async by design**: it returns HTTP 200 before pipeline processing starts. This prevents ElevenLabs from disabling the webhook due to timeouts (~30s limit). Pipeline processing runs via FastAPI `BackgroundTasks` wrapped in `_process_interview_safe()` which catches and logs all exceptions.

### Dependency Injection

All services and agents are instantiated in `main.py:lifespan()` and stored on `app.state`. Route handlers access them via dependency functions in `api/deps.py` (e.g., `get_pipeline(request)`). No global singletons — everything flows through FastAPI's DI.

### Key Concepts

- **Evidence**: Verbatim quote + English interpretation structured as Factor → Mechanism → Outcome
- **Proposition**: A causal hypothesis with confidence score and status (`untested`, `exploring`, `confirmed`, `challenged`, `saturated`, `weak`, `merged`), built from evidence across interviews
- **Convergence**: System auto-switches from divergent mode (exploring hypotheses) to convergent mode (confirming them) when `convergence_score > 0.6` and `novelty_rate < 0.15`
- **Script Safety** (`services/script_safety.py`): Validates generated scripts — detects personal references, topic drift; sanitizes or falls back to safe templates

### Error Handling & Data Safety

`pipeline.process_interview()` uses phased try/except to guarantee data persistence:

1. **Interview saved first** — raw transcript written to disk before any LLM calls
2. **Analysis phase** — on failure, interview + conversation_id saved, exception re-raised
3. **Script generation phase** — on failure, analysis results saved, exception re-raised
4. **ElevenLabs sync** — on failure, marks `sync_pending=True` and continues (non-fatal)

The webhook wrapper `_process_interview_safe()` catches all exceptions from the pipeline and logs them, ensuring background task failures never surface as unhandled errors.

### Persistence

File-based JSON storage: `data/projects/{project_id}/` containing `project.json`, `interviews/`, and `scripts/` subdirectories. Managed by `services/project_service.py`. All project state lives in `models/project.py:ProjectState` (Pydantic model).

**Note**: On ephemeral filesystems (e.g., Koyeb), data is lost on redeploy. `DATA_DIR` should point to a persistent volume in production.

### Real-Time Updates

SSE streaming via `services/sse_manager.py` → consumed by React dashboard (`dashboard/`). Event types include `new_evidence`, `proposition_updated`, `script_updated`, `prompt_sanitized`, `topic_redirect_applied`, `report_stale`, `project_stats`, `visualization_model_ready`.

### Prompt Files

System prompts are plain text in `prompts/` directory with per-language variants in `prompts/en/` and `prompts/ru/`, loaded at runtime by `agents/prompt_loader.py`. The interviewer prompt uses `{placeholder}` template substitution for dynamic content. Prompts are cached per-language in agent instances (`_system_prompts`, `_interviewer_prompts` dicts) — restart the server to pick up prompt file changes.

## Code Conventions

- All Pydantic models use `ConfigDict(extra="ignore")` to tolerate extra fields from LLM JSON output.
- Config is a frozen dataclass (`config.py:Settings`) loaded once via `@lru_cache`. Per-agent model overrides: `DESIGNER_MODEL`, `ANALYST_MODEL`, `SYNTHESIZER_MODEL` env vars fall back to `MISTRAL_MODEL`.
- Tests mock `LLMClient.chat_json` / `LLMClient.chat` to avoid real API calls. See existing tests for patterns.
- Fake designer mocks must accept `language` parameter: `build_interviewer_prompt(self, script, language="en")`.
- Logging uses `logging.getLogger(__name__)` — no print statements. `logging.basicConfig()` is configured in `main.py` at module level (INFO level, timestamped format). Key modules with logging: `api/routes_webhook.py`, `services/pipeline.py`, `services/elevenlabs_service.py`.

## Environment Variables

Required keys: `MISTRAL_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_AGENT_ID`. See `.env.example` for all variables including per-agent model overrides, convergence thresholds, and retry settings.

## API Structure

Three routers mounted in `main.py` under `/api`:
- `api/routes_projects.py` — project CRUD, start, simulate, report, evidence/propositions access
- `api/routes_webhook.py` — ElevenLabs post-call webhook receiver (async: returns 200 immediately, processes via BackgroundTasks)
- `api/routes_stream.py` — SSE event stream per project

## Deployment (Koyeb)

App: `working-sheryl`, service: `mistral-hackaton`.
Production URL: `https://working-sheryl-grounded-7ef2e4c4.koyeb.app`

### Koyeb CLI Commands

```bash
# Check service status and deployment history
koyeb service describe working-sheryl/mistral-hackaton

# View runtime logs (stdout/stderr from the app)
koyeb service logs working-sheryl/mistral-hackaton --type runtime

# View build logs (pip install, buildpack output)
koyeb service logs working-sheryl/mistral-hackaton --type build

# Filter logs (pipe to grep)
koyeb service logs working-sheryl/mistral-hackaton --type runtime | grep -i "POST\|error\|webhook"

# Redeploy (pulls latest commit from GitHub)
koyeb service redeploy working-sheryl/mistral-hackaton

# Exec into running instance
koyeb service exec working-sheryl/mistral-hackaton -- sh

# List instances
koyeb instance list
```

### ElevenLabs Agent Configuration

The ElevenLabs conversational agent uses Mistral as a custom LLM (`custom-llm` mode). Key settings managed via ElevenLabs API:
- `custom_llm.url`: `https://api.mistral.ai/v1`
- `custom_llm.model_id`: `mistral-large-latest`
- `custom_llm.api_key`: stored as ElevenLabs secret (not in our repo)
- `max_tokens`: must be a positive integer (e.g., 1000). `-1` causes Mistral API errors.
- `reasoning_effort`: must be `null`. Setting it to any value (e.g., `"none"`) causes `custom_llm_error` because ElevenLabs passes it to Mistral, which doesn't support it.
- `temperature`: `0.0`

To inspect/update agent config via API:
```bash
# Get agent config
curl -s "https://api.elevenlabs.io/v1/convai/agents/$ELEVENLABS_AGENT_ID" \
  -H "xi-api-key: $ELEVENLABS_API_KEY" | python3 -m json.tool

# Patch agent config (example: fix max_tokens and reasoning_effort)
curl -X PATCH "https://api.elevenlabs.io/v1/convai/agents/$ELEVENLABS_AGENT_ID" \
  -H "xi-api-key: $ELEVENLABS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation_config":{"agent":{"prompt":{"max_tokens":1000,"reasoning_effort":null}}}}'
```

### Troubleshooting

- **`custom_llm_error: Failed to generate response from custom LLM`**: Check ElevenLabs agent config — `reasoning_effort` must be `null` (not `"none"`), `max_tokens` must be a positive integer (not `-1`). Use the PATCH commands above to fix. Also verify the Mistral API key secret is valid.
- **Webhook disabled by ElevenLabs**: If "Auto disabled due to repeated failures" appears in ElevenLabs → Developers → Webhooks, re-enable manually. The webhook must return HTTP 200 within ~30s (our code returns instantly via BackgroundTasks).
- **Data lost after redeploy**: Koyeb uses ephemeral filesystem. All project data in `data/` is wiped on each deploy. Use `scripts/import_transcripts.py` to re-import from saved transcripts.
- **Deployment ERROR "Failed to get SHA"**: Usually a transient GitHub connectivity issue. Retry with `koyeb service redeploy`.
- **No pipeline logs visible**: Ensure `logging.basicConfig()` is in `main.py`. Without it, `getLogger(__name__)` defaults to WARNING level and INFO messages are silently dropped.

## Training Pipeline (Optional)

`training/` contains a fine-tuning pipeline for Mistral models: transcribe → normalize speakers → extract examples → anonymize → format JSONL → launch fine-tune job. Each step has corresponding tests in `tests/test_training_*.py`.
