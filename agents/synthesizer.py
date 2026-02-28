from __future__ import annotations

import json

from agents.llm_client import LLMClient
from agents.prompt_loader import load_prompt
from models.project import ProjectState


class SynthesizerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("synthesizer_system.txt")

    async def synthesize(self, project: ProjectState) -> str:
        payload = {
            "research_question": project.research_question,
            "evidence": [e.model_dump(mode="json") for e in project.evidence_store],
            "propositions": [p.model_dump(mode="json") for p in project.proposition_store],
            "metrics": project.metrics.model_dump(mode="json"),
            "interviews": len(project.interview_store),
            "script_versions": len(project.script_versions),
        }
        return await self.llm.chat(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
