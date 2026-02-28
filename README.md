# Eidetic MVP

Autonomous qualitative research loop:

`webhook -> analyst -> designer -> ElevenLabs prompt update -> JSON storage -> SSE`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Demo UI: `http://localhost:8000/ui`

## Key endpoints

- `POST /api/projects`
- `GET /api/projects/cards`
- `POST /api/projects/{id}/start`
- `POST /api/projects/{id}/finish`
- `GET /api/projects/{id}/report`
- `GET /api/projects/{id}/qrcode`
- `POST /api/webhook/elevenlabs`
- `GET /api/projects/{id}`
- `GET /api/projects/{id}/stream`
- `POST /api/projects/{id}/synthesize`

## Local test helper

Simulate interview without ElevenLabs:

```bash
curl -X POST http://localhost:8000/api/projects/demo/simulate \
  -H "Content-Type: application/json" \
  -d '{"transcript":"User: Time pressure helped us focus"}'
```

Inspect project summary:

```bash
python scripts/show_project.py demo
```

## Tests

```bash
pytest -q
```
