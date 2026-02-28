# Eidetic — Setup & Configuration Guide

## Prerequisites

- Python 3.12+
- Node.js 18+ (for dashboard)
- Git
- A Mistral API key with access to `mistral-large-latest`
- An ElevenLabs account with Conversational AI credits
- A Koyeb account (for deployment)

---

## 1. Repository Setup

```bash
git clone https://github.com/shipaleks/mistral-hackaton.git
cd mistral-hackaton
```

### Project Structure

Create the following directory structure:

```
eidetic/
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
├── .env
├── .env.example
├── .gitignore
├── agents/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── designer.py
│   ├── analyst.py
│   └── synthesizer.py
├── models/
│   ├── __init__.py
│   ├── evidence.py
│   ├── proposition.py
│   ├── interview.py
│   ├── script.py
│   └── project.py
├── services/
│   ├── __init__.py
│   ├── elevenlabs_service.py
│   ├── project_service.py
│   ├── pipeline.py
│   └── sse_manager.py
├── api/
│   ├── __init__.py
│   ├── routes_projects.py
│   ├── routes_webhook.py
│   └── routes_stream.py
├── prompts/
│   ├── designer_system.txt
│   ├── analyst_system.txt
│   ├── synthesizer_system.txt
│   └── interviewer_base.txt
├── dashboard/
│   └── (React app — created via Vite or CRA)
├── training/
│   ├── transcribe.py
│   ├── extract_examples.py
│   ├── anonymize.py
│   ├── format_jsonl.py
│   └── finetune.py
└── data/
    └── projects/
        └── .gitkeep
```

---

## 2. Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### requirements.txt

```
fastapi==0.115.*
uvicorn[standard]==0.34.*
httpx==0.28.*
python-dotenv==1.0.*
pydantic==2.10.*
sse-starlette==2.2.*
```

---

## 3. Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

### .env.example

```bash
# ============================================
# Mistral AI
# ============================================
MISTRAL_API_KEY=your_mistral_api_key_here

# ============================================
# ElevenLabs
# ============================================
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_AGENT_ID=            # Set after creating agent in ElevenLabs dashboard

# ============================================
# Application
# ============================================
APP_BASE_URL=http://localhost:8000       # Change to Koyeb URL after deploy
DATA_DIR=./data/projects
DEFAULT_PROJECT_ID=hackathon-demo

# ============================================
# LLM Configuration
# ============================================
# All agents use mistral-large-latest by default.
# Override per-agent if needed (e.g., for latency testing):
# INTERVIEWER_MODEL=mistral-medium-latest
# ANALYST_MODEL=mistral-large-latest
# DESIGNER_MODEL=mistral-large-latest
# SYNTHESIZER_MODEL=mistral-large-latest

# ============================================
# Convergence Thresholds
# ============================================
CONVERGENCE_SCORE_THRESHOLD=0.6
NOVELTY_RATE_THRESHOLD=0.15
MERGE_OVERLAP_THRESHOLD=0.6
PRUNE_CONFIDENCE_THRESHOLD=0.15
PRUNE_MIN_INTERVIEWS=3

# ============================================
# Interview Constraints
# ============================================
MAX_INTERVIEW_DURATION_MINUTES=10
MAX_PROPOSITIONS_IN_SCRIPT=8

# ============================================
# Optional: Fine-tuning & Logging
# ============================================
# WANDB_API_KEY=your_wandb_key_here
```

### .gitignore

```
.env
venv/
__pycache__/
data/projects/*/
*.pyc
node_modules/
dashboard/build/
dashboard/dist/
.DS_Store
```

---

## 4. Verify API Access

### Mistral API

```bash
curl -s -X POST https://api.mistral.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -d '{
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Reply with just: OK"}],
    "max_tokens": 10
  }' | python -m json.tool
```

Expected: JSON response with `choices[0].message.content` = "OK"

### ElevenLabs API

```bash
curl -s https://api.elevenlabs.io/v1/convai/agents \
  -H "xi-api-key: $ELEVENLABS_API_KEY" | python -m json.tool
```

Expected: JSON with list of your agents (may be empty).

---

## 5. ElevenLabs Agent Setup (Manual — Dashboard)

This is done ONCE before running the code pipeline.

### Step-by-step in ElevenLabs Dashboard

1. Go to https://elevenlabs.io/app/conversational-ai
2. Click **Create Agent** → **Blank template**
3. Configure **Agent** tab:
   - **Name**: `Eidetic Interviewer`
   - **First message**: 
     ```
     Hey! I'm Sasha, an AI research assistant. I'm studying what people think about this hackathon. Could you share any thoughts — literally anything that comes to mind?
     ```
   - **System prompt**: Paste the content of `prompts/interviewer_base.txt` with placeholder sections replaced by generic text (this will be overwritten by PATCH API during runtime)
   
4. Configure **LLM**:
   - Select **Custom LLM** from dropdown
   - **Server URL**: `https://api.mistral.ai/v1`
   - **Model ID**: `mistral-large-latest`
   - Add **Secret**: 
     - Go to Secrets section → Add Secret
     - Type: Custom LLM
     - Value: your Mistral API key
     - Save
   - Select the secret in the LLM configuration

5. Configure **Voice**:
   - Pick a voice from the library (recommendations: "Rachel", "Josh", or any neutral/friendly voice)
   - Language: English
   - Add additional languages: French, Japanese, Russian

6. Configure **System Tools**:
   - Enable **end_call** (agent can end conversation)
   - Enable **language_detection** (agent switches language automatically)

7. Configure **Security** tab:
   - Enable **System prompt** override ← CRITICAL for PATCH API to work
   - Enable **First message** override (optional but useful)

8. Configure **Post-call Webhook**:
   - URL: `https://{your-koyeb-app-url}/api/webhook/elevenlabs`
   - Enable: transcript data
   - (Set this AFTER deploying to Koyeb — use localhost:8000 for local testing with ngrok)

9. **Save** the agent
10. Copy the **Agent ID** from the URL or agent settings → put in `.env` as `ELEVENLABS_AGENT_ID`

### Verify PATCH works

```bash
curl -X PATCH "https://api.elevenlabs.io/v1/convai/agents/$ELEVENLABS_AGENT_ID" \
  -H "xi-api-key: $ELEVENLABS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_config": {
      "agent": {
        "prompt": {
          "prompt": "You are a test agent. Say hello and end the call."
        }
      }
    }
  }'
```

Expected: 200 OK. Go to ElevenLabs dashboard → verify the system prompt has changed.

**Restore original prompt after testing** by PATCHing back the real prompt.

### Get Talk-to Link

The public link for respondents to call the agent:
```
https://elevenlabs.io/app/talk-to/{ELEVENLABS_AGENT_ID}
```

---

## 6. Local Development

### Run Backend

```bash
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at: http://localhost:8000
API docs (Swagger): http://localhost:8000/docs

### Expose Localhost for Webhooks (for local testing)

ElevenLabs post-call webhooks need a public URL. Use ngrok:

```bash
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`) and set it as the webhook URL in ElevenLabs dashboard.

### Run Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard available at: http://localhost:5173 (Vite default)

---

## 7. Deployment to Koyeb

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build React dashboard (if included in same container)
# RUN cd dashboard && npm install && npm run build

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deploy Steps

1. Push code to GitHub: `git push origin main`
2. Go to https://app.koyeb.com
3. Create new App → GitHub → select `shipaleks/mistral-hackaton`
4. Build method: **Docker**
5. Region: **Frankfurt** (fra)
6. Instance: **Nano** (sufficient for hackathon)
7. Environment variables: add all from `.env`
8. Deploy

### After Deployment

1. Copy the Koyeb app URL (e.g., `https://eidetic-xyz.koyeb.app`)
2. Update `.env`: `APP_BASE_URL=https://eidetic-xyz.koyeb.app`
3. Update ElevenLabs webhook URL to: `https://eidetic-xyz.koyeb.app/api/webhook/elevenlabs`
4. Verify: call the agent → check Koyeb logs for webhook receipt

---

## 8. Creating a Research Project

### Via API

```bash
# Create project
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "id": "hackathon-demo",
    "research_question": "What is your experience with this hackathon so far?",
    "initial_angles": ["organization", "food", "mentors", "difficulty", "team formation", "time pressure"]
  }'

# Start project (generates initial propositions + script, updates ElevenLabs agent)
curl -X POST http://localhost:8000/api/projects/hackathon-demo/start

# Check project state
curl http://localhost:8000/api/projects/hackathon-demo | python -m json.tool

# View current propositions
curl http://localhost:8000/api/projects/hackathon-demo/propositions | python -m json.tool

# View script versions
curl http://localhost:8000/api/projects/hackathon-demo/scripts | python -m json.tool

# Generate final report
curl -X POST http://localhost:8000/api/projects/hackathon-demo/synthesize

# Delete project
curl -X DELETE http://localhost:8000/api/projects/hackathon-demo
```

### Via Dashboard

Open `http://localhost:5173` (or Koyeb URL) → "New Project" → fill in research question → Start.

---

## 9. Fine-Tuning Pipeline Setup (Optional)

### Prerequisites
- Interview audio files (`.mp3`, `.m4a`, or `.webm` from Zoom)
- Place them in `training/audio/` directory

### Steps

```bash
# 1. Transcribe with Voxtral (diarization enabled)
python training/transcribe.py --input-dir training/audio/ --output-dir training/transcripts/

# 2. Extract training examples (good interviewer questions)
python training/extract_examples.py --input-dir training/transcripts/ --output training/examples_raw.jsonl

# 3. Anonymize (remove PII + all Yandex references)
python training/anonymize.py --input training/examples_raw.jsonl --output training/examples_clean.jsonl

# 4. Format for Mistral fine-tuning API
python training/format_jsonl.py --input training/examples_clean.jsonl --output training/finetune_data.jsonl

# 5. Launch fine-tuning job
python training/finetune.py --data training/finetune_data.jsonl --model mistral-small-latest
```

Fine-tuning takes ~30-60 minutes. The resulting model ID can be used as `INTERVIEWER_MODEL` in `.env`.

---

## 10. Troubleshooting

### ElevenLabs agent doesn't respond
- Check: Is Custom LLM secret set correctly?
- Check: Is the Mistral API key valid?
- Check: Is `mistral-large-latest` accessible with your key?
- Try: Switch to a natively supported LLM (e.g., GPT-4o) temporarily to isolate the issue

### Webhook not arriving
- Check: Is the webhook URL correct in ElevenLabs dashboard?
- Check: Is the Koyeb app running? (check logs)
- Check: Is the endpoint `/api/webhook/elevenlabs` returning 200?
- Try: Use ngrok for local testing first

### PATCH returns 403
- Check: Is "System prompt override" enabled in ElevenLabs Security settings?
- Check: Is the ElevenLabs API key correct?

### Mistral responses are slow (>2 sec TTFT for interviewer)
- This is expected for `mistral-large-latest` in voice mode
- If unbearable: change `INTERVIEWER_MODEL` to `mistral-medium-latest` or `mistral-small-latest`
- Does NOT affect analyst/designer/synthesizer (they run async, latency doesn't matter)

### Dashboard not updating
- Check: Is SSE endpoint (`/api/projects/{id}/stream`) connected? (browser dev tools → Network → EventStream)
- Check: Is the pipeline emitting events? (check backend logs)
- Try: Hard refresh the dashboard page

### JSON parse errors from LLM
- Mistral Large occasionally returns malformed JSON
- The LLM client should have retry logic (3 attempts with temperature increase)
- If persistent: add `response_format: {"type": "json_object"}` to the API call (Mistral supports this)
