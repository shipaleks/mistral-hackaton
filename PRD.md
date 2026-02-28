# Eidetic: Autonomous Qualitative Research System

## Product Requirements Document

### Project Overview

**Eidetic** is an autonomous AI-powered qualitative research system that conducts adaptive voice interviews, extracts causal propositions from conversations, and evolves its interview strategy based on accumulating evidence — converging from chaos to insight without human intervention.

Unlike simple AI interviewers that follow a static script, Eidetic implements an **automated grounded theory methodology**: it generates hypotheses, tests them across interviews, discovers new patterns, merges and prunes propositions, and adapts its questions in real-time — mimicking how an expert researcher thinks between interviews, but faster.

### Hackathon Context

- **Event**: Mistral AI Worldwide Hackathon 2026, Paris (in-person)
- **Dates**: February 28 – March 1, 2026 (48 hours)
- **Team**: Solo (Aleks)
- **Prize Tracks**:
  - Main track: Top 3 in Paris ($500–$1,500 + Mistral credits)
  - Global winner ($10,000 + $15,000 credits + hiring opportunity)
  - **Best Use of ElevenLabs** ($2,000 in credits) — strong fit
  - Best Use of Agent Skills (Reachy Mini robot)
- **Submission**: Code repository + functional demo + pitch presentation
- **Budget**: ~150–200€ for API costs

### Demo Scenario

**Research Question**: "What is your experience with this hackathon so far?"

**Setup**: Eidetic conducts up to 10 voice interviews (10 min each) with hackathon participants during the event. Interviews run overnight Saturday → Sunday morning. The system autonomously evolves its interview guide between conversations.

**Demo for Judges**: Recorded interviews + live dashboard showing:
- Knowledge graph growing over time
- Propositions emerging, being confirmed/challenged
- Interview scripts evolving between conversations
- Final synthesized report

**Languages**: Multilingual — the agent automatically detects and responds in the respondent's language. The hackathon runs in 7 cities (Paris, London, New York, San Francisco, Tokyo, Singapore, Sydney) + online, so we support:
- **English** (all cities, primary)
- **French** (Paris)
- **Japanese** (Tokyo)
- **Russian** (online participants, testing)

This is a key differentiator: respondents share richer, more nuanced insights when interviewed in their native language. The system extracts evidence and builds propositions cross-lingually — all internal analysis, hypotheses, and reports are always in English regardless of interview language.

---

## System Architecture

### Three Agents + One Pipeline

```
DESIGNER → INTERVIEWER → ANALYST → (loop) → DESIGNER
                                         ↘ SYNTHESIZER (final)
```

#### Agent 1: DESIGNER (Mistral Large)
- **Input**: Research question + current Proposition Store + Evidence Store
- **Output**: Interview Script (system prompt for voice agent)
- **When**: At project start + after each interview analysis completes
- **Key behavior**: Assigns instructions per proposition (EXPLORE / VERIFY / CHALLENGE / SATURATED), prioritizes questions by confidence gap and novelty

#### Agent 2: INTERVIEWER (ElevenLabs + Mistral Large via Custom LLM)
- **Input**: Interview Script (as system prompt)
- **Output**: Voice conversation → transcript (via post-call webhook)
- **When**: Continuously, as respondents connect
- **Key behavior**: Friendly, open-ended interviewing. Never leads. Never asks about interfaces/screens. Follows script priorities but pursues unexpected threads when they emerge.

#### Agent 3: ANALYST (Mistral Large)
- **Input**: Transcript + current Evidence Store + Proposition Store
- **Output**: Updated Evidence Store + Proposition Store
- **When**: Automatically triggered by post-call webhook
- **Key behavior**: Extracts evidence, maps to propositions, detects orphans, generates new propositions, retroactive scan of all evidence against new propositions, recalculates confidence, merges/prunes

#### Agent 4: SYNTHESIZER (Mistral Large)
- **Input**: Final Proposition Store + Evidence Store
- **Output**: Research report with core findings, supporting evidence, methodology description
- **When**: Manually triggered, or when convergence thresholds met

### Data Training Pipeline (Pre-hackathon prep)

Separate from the main loop. Uses existing interview recordings to generate fine-tuning data:

```
Audio files (Zoom) → Voxtral Transcribe (diarization) → Mistral Large (extract training examples + anonymize) → JSONL → Fine-tune Mistral Small
```

**Anonymization requirement**: All personal data AND any Yandex-related content must be stripped from transcripts before use.

---

## Data Model

### Evidence

An atomic observation extracted from an interview.

```json
{
  "id": "E001",
  "interview_id": "INT_003",
  "quote": "We actually worked better because we didn't know each other",
  "interpretation": "Stranger teams may have less overhead due to absence of pre-existing dynamics",
  "factor": "team composition with strangers",
  "mechanism": "absence of pre-existing social dynamics",
  "outcome": "reduced communication overhead",
  "tags": ["team_formation", "productivity", "social_dynamics"],
  "language": "en",
  "timestamp": "2026-02-28T23:15:00Z"
}
```

Each evidence is structured as **Factor → Mechanism → Outcome** (simplified causal model). This is the knowledge representation framework — not the full Strauss-Corbin paradigm model (too rigid for LLM extraction), but enough to generate testable causal claims.

**Cross-lingual example**: A Japanese respondent says:
```json
{
  "id": "E015",
  "interview_id": "INT_007",
  "quote": "チームメンバーを知らなかったから、逆に政治がなくてよかった",
  "interpretation": "Unfamiliarity between team members eliminated pre-existing politics, which was perceived as beneficial",
  "factor": "team composition with strangers",
  "mechanism": "absence of pre-existing social dynamics",
  "outcome": "reduced interpersonal friction",
  "tags": ["team_formation", "social_dynamics", "stranger_teams"],
  "language": "ja",
  "timestamp": "2026-03-01T02:30:00Z"
}
```
This evidence maps to the SAME proposition P002 as the English quote from E001 — the system builds knowledge cross-lingually.

### Proposition

A causal claim derived from clustered evidence. A proposition is a **meta-object over evidence**, not a node at the same level.

```json
{
  "id": "P002",
  "factor": "Team formation with strangers",
  "mechanism": "Absence of established trust and work patterns",
  "outcome": "Communication overhead reduces productivity",
  "confidence": 0.25,
  "status": "challenged",
  "supporting_evidence": ["E005"],
  "contradicting_evidence": ["E001", "E012"],
  "first_seen_interview": 1,
  "last_updated_interview": 3,
  "interviews_without_new_evidence": 0
}
```

**Confidence** = `supporting / (supporting + contradicting)` weighted by recency. Not a precise statistical measure — a heuristic for prioritization.

**Status values**:
- `untested` — no evidence yet (just generated)
- `exploring` — some evidence, still gathering
- `confirmed` — confidence > 0.7, multiple supporting evidence
- `challenged` — significant contradicting evidence
- `saturated` — confidence > 0.8 AND no new evidence for 2+ interviews
- `weak` — confidence < 0.15 after 3+ interviews → removed from active script
- `merged` — absorbed into another proposition

### Interview Script

The dynamic prompt that controls the INTERVIEWER agent.

```json
{
  "version": 7,
  "generated_after_interview": "INT_006",
  "research_question": "What is your experience with this hackathon?",
  "opening_question": "Hey! Could you share any thoughts about the hackathon so far?",
  "sections": [
    {
      "proposition_id": "P001",
      "priority": "medium",
      "instruction": "VERIFY",
      "main_question": "How did you approach the time constraint?",
      "probes": ["What did you decide to cut?", "How did that feel?"],
      "context": "3 supporting, 1 contradicting. Look for nuance in tradeoff decisions."
    }
  ],
  "closing_question": "What surprised you most about this experience?",
  "wildcard": "Is there anything important I haven't asked about?",
  "mode": "convergent",
  "convergence_score": 0.62,
  "novelty_rate": 0.18
}
```

### Project

Isolated container for one research study.

```json
{
  "id": "hackathon-demo",
  "research_question": "What is your experience with this hackathon?",
  "created_at": "2026-02-28T18:00:00Z",
  "evidence_store": [],
  "proposition_store": [],
  "interview_store": [],
  "script_versions": [],
  "elevenlabs_agent_id": "agent_xxx"
}
```

All data stored as JSON files on disk, organized per project. No database needed for hackathon scale.

```
/data/projects/
  /hackathon-demo/
    project.json
    evidence_store.json
    proposition_store.json
    interviews/
      INT_001.json
      INT_002.json
    scripts/
      script_v1.json
      script_v2.json
```

---

## Convergence Engine

The system operates in two modes to prevent hypothesis explosion:

### Divergent Mode (first ~5 interviews)
- Generate propositions aggressively from orphan evidence
- Low merge/prune threshold
- All propositions marked as EXPLORE in script
- Goal: cover the space

### Convergent Mode (after threshold)
Activated when `convergence_score > 0.6` AND `novelty_rate < 0.15`:

```
convergence_score = confirmed_propositions / total_active_propositions
novelty_rate = new_evidence_creating_new_propositions / total_new_evidence
```

**Merge**: Two propositions with >60% evidence overlap → combine into a more general proposition. The specific ones become supporting evidence for the general one.

**Prune**: Proposition with confidence < 0.15 after being tested in 3+ interviews → status: `weak`, removed from active script.

**Subsume**: When one proposition is a special case of another → the specific one becomes evidence for the general one.

**Core Proposition Election**: After 7+ interviews, identify 1-3 propositions with highest confidence AND most connections to other propositions. These are "core findings". All other propositions organize around them.

**Visual on dashboard**: Starts as chaotic cloud → clusters form → clusters merge → final 3-5 clear "islands" of knowledge.

---

## Event-Driven Pipeline

The system is **not batch-based** (no waves). It's event-driven:

```
Interview N ends
  → post-call webhook fires (transcript arrives)
  → ANALYST processes transcript (30-60 sec)
  → Evidence Store updated
  → Proposition Store updated (new props, confidence recalc, merge/prune)
  → DESIGNER generates new script
  → PATCH /v1/convai/agents/{id} with new system prompt
  → Next respondent gets updated agent
```

If two interviews happen in parallel — each gets the script version available at its start time. The system updates as fast as data arrives. No artificial synchronization barriers.

### Retroactive Scan

When a new proposition is generated from Interview N, the ANALYST checks ALL existing evidence (from interviews 1..N-1) against it. This catches evidence that was "orphaned" at extraction time but actually supports the new proposition.

**Scale**: 10 interviews × 20 evidence each = 200 evidence objects × ~100 tokens = 20K tokens. One Mistral Large call. Even at 30 interviews (750 evidence), fits in a single call. Cost: ~$0.04.

---

## ElevenLabs Integration

### Agent Configuration

```
LLM: Custom LLM
Server URL: https://api.mistral.ai/v1
Model ID: mistral-large-latest
API Key: [stored as ElevenLabs Secret]
Voice: [selected from ElevenLabs library — neutral, friendly]
Languages: en, fr, ja, ru
System tools: end_call (enabled), language_detection (enabled)
Post-call webhook: https://{koyeb-app-url}/api/webhook/elevenlabs
Overrides: system prompt override enabled (for PATCH updates)
```

### Script Updates via PATCH API

After each analysis cycle, the backend calls:

```
PATCH https://api.elevenlabs.io/v1/convai/agents/{agent_id}
Authorization: Bearer {elevenlabs_api_key}
Content-Type: application/json

{
  "conversation_config": {
    "agent": {
      "prompt": {
        "prompt": "<full updated system prompt with new script>"
      }
    }
  }
}
```

This updates the agent for all future conversations without affecting active ones.

### Post-Call Webhook Payload

ElevenLabs sends transcript data to our backend after each conversation ends. Key fields:
- `conversation_id`
- `transcript` (full text with speaker labels and timestamps)
- `analysis` (if configured — we may skip this and do our own)
- `metadata` (duration, agent_id)

---

## Tech Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| LLM (all agents) | Mistral Large (`mistral-large-latest`) | Hackathon requirement. Best quality available. |
| Voice platform | ElevenLabs Conversational AI | Hackathon sponsor. Best voice use case prize. |
| Backend | Python (FastAPI) | Fast to build, good async support |
| Frontend/Dashboard | React + d3.js (force-directed graph) | Beautiful graph visualization |
| Deployment | Koyeb | Mistral acquired Koyeb — good signal for judges |
| Data storage | JSON files on disk | Sufficient for hackathon scale |
| Transcription (training) | Voxtral Transcribe V2 | Mistral product, diarization support |
| Fine-tuning (optional) | Mistral Fine-tuning API | Distillation demo |

### LLM-Agnostic Design

All LLM calls go through a unified wrapper:

```python
# config.py
LLM_CONFIG = {
    "designer": {"provider": "mistral", "model": "mistral-large-latest"},
    "analyst": {"provider": "mistral", "model": "mistral-large-latest"},
    "synthesizer": {"provider": "mistral", "model": "mistral-large-latest"},
    "interviewer": {"provider": "mistral", "model": "mistral-large-latest"},
}
```

Post-hackathon: changing `provider` to `"anthropic"` or `"openai"` switches the entire system. All prompts are provider-agnostic (no Mistral-specific syntax).

---

## API Endpoints (Backend)

```
POST   /api/projects                    — Create new research project
GET    /api/projects/{id}               — Get project state
POST   /api/projects/{id}/start         — Generate initial script, create ElevenLabs agent
POST   /api/webhook/elevenlabs          — Receive post-call transcript
GET    /api/projects/{id}/evidence      — Get all evidence
GET    /api/projects/{id}/propositions  — Get all propositions
GET    /api/projects/{id}/scripts       — Get all script versions
POST   /api/projects/{id}/synthesize    — Trigger final report
GET    /api/projects/{id}/stream        — SSE stream for dashboard updates
DELETE /api/projects/{id}               — Delete project and all data
```

---

## Dashboard

### Visualization: Force-Directed Knowledge Graph

**Nodes**: Evidence objects (small circles, color-coded by interview)

**Edges**: Semantic similarity between evidence (threshold > 0.7)

**Overlays**: Propositions as translucent regions (convex hulls) around their supporting evidence clusters. Color indicates confidence (red = low, yellow = medium, green = high).

**Animations**: After each interview analysis:
- New evidence nodes appear (fade in)
- New edges form (animated lines)
- Proposition regions expand/contract
- Merged propositions animate together
- Pruned propositions fade to gray

**Tech**: React + d3.js force-directed graph. Backend pushes updates via Server-Sent Events (SSE). Each analysis step emits incremental events:
```
event: new_evidence
data: {"id": "E042", "tags": [...], "position": {...}}

event: proposition_updated
data: {"id": "P002", "confidence": 0.25, "status": "challenged"}

event: proposition_merged
data: {"source": "P003", "target": "P007", "result": "P003'"}
```

### Additional Dashboard Panels

- **Convergence Meter**: Bar showing divergent → convergent progression
- **Script Diff**: Side-by-side comparison of script versions (what changed)
- **Interview Timeline**: List of completed interviews with key extractions
- **Proposition Table**: Sortable list of all propositions with confidence, status, evidence count

---

## Cost Estimates

### Mistral API (30 interviews scenario)

| Operation | Calls | Input tokens | Output tokens | Cost |
|-----------|-------|-------------|--------------|------|
| Designer (initial + updates) | 30 | ~15K each | ~3K each | ~$0.36 |
| Interviewer (voice, via ElevenLabs) | 30 | ~10K each | ~5K each | ~$0.38 |
| Analyst (per interview) | 30 | ~30K each (growing) | ~5K each | ~$0.68 |
| Retroactive scans | ~50 | ~50K each | ~2K each | ~$1.40 |
| Synthesizer (final) | 1 | ~100K | ~10K | ~$0.07 |
| **Total Mistral** | | | | **~$3–5** |

### Voxtral Transcription (training data)
- 100 recordings × 20 min = 2000 min × $0.003/min = **$6**

### ElevenLabs
- Main cost item. Creator plan credits should cover demo. Backup: ~$20–50 for additional credits.

**Total estimated cost: $30–60 (well within 150–200€ budget)**

---

## Fine-Tuning Pipeline (Optional, Time-Permitting)

### Goal
Distill Mistral Large interviewing quality into Mistral Small for lower latency voice conversations.

### Data Pipeline
1. Existing interview audio (Zoom) → Voxtral Transcribe V2 (diarization, ~37 sec/interview)
2. Mistral Large extracts training examples: each interviewer turn = one example (context → question)
3. Quality filter: only 4+/5 rated examples kept
4. Anonymization: strip all personal data and Yandex references
5. Format to JSONL for Mistral fine-tuning API
6. Fine-tune Mistral Small (~30-60 min)
7. Compare base vs fine-tuned in parallel test

### If Time Allows
- Log metrics to W&B for cross-track submission
- Show comparison in demo: "Here's base model, here's fine-tuned"

---

## Judging Criteria Alignment

| Criterion | Our Approach |
|-----------|-------------|
| **Technical Implementation** | Multi-agent system with shared knowledge store, event-driven pipeline, dynamic prompt generation, retroactive evidence scanning, convergence engine |
| **Creativity / Uniqueness** | No one has automated grounded theory with evolving voice interviews. System interviews in 4 languages and builds cross-lingual proposition graphs — a Japanese quote and a French quote support the same English-language hypothesis. Generates questions humans wouldn't think to ask by cross-referencing all interviews simultaneously. |
| **Future Potential** | Real B2B product. Replaces weeks of qualitative research with hours. Self-hosted Mistral for sensitive enterprise data. |
| **Pitch Quality** | Live demo with real data from hackathon participants. Knowledge graph growing on screen. Script evolution visible. |

### One-Sentence Pitch

> "Eidetic conducts AI voice interviews in any language that get smarter after every conversation — automatically building and testing causal theories about user behavior, converging from scattered observations to validated insights."

### Key Differentiators

1. **Self-evolving hypotheses**: System generates questions human researchers might never ask because it sees patterns across ALL interviews simultaneously.

2. **Multilingual cross-analysis**: A Japanese respondent in Tokyo and a French respondent in Paris contribute to the SAME proposition graph. Evidence quotes are preserved in original language, all analysis and hypotheses are in English. This is impossible with human researchers without expensive translation workflows.

3. **Example**: Interview #1 (English) mentions "don't trust new apps", Interview #2 (Japanese) mentions "いつもレビューを読む" (always read reviews) — system connects [trust, social_proof] across languages, generates English-language hypothesis "social proof is key trust mechanism", Interview #3 (French) gets the new question in French.
