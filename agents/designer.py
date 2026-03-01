from __future__ import annotations

import json
from typing import Any

from agents.llm_client import LLMClient
from agents.prompt_loader import load_prompt
from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection


_VALID_PROPOSITION_STATUSES = {
    "untested",
    "exploring",
    "confirmed",
    "challenged",
    "saturated",
    "weak",
    "merged",
}


class DesignerAgent:
    def __init__(self, llm: LLMClient, max_sections: int = 8):
        self.llm = llm
        self.max_sections = max_sections
        self.system_prompt = load_prompt("designer_system.txt")
        self.interviewer_base_prompt = load_prompt("interviewer_base.txt")

    async def generate_initial_script(
        self, research_question: str, initial_angles: list[str] | None = None
    ) -> tuple[list[Proposition], InterviewScript]:
        payload = {
            "task": "Generate initial propositions and first interview script",
            "research_question": research_question,
            "initial_angles": initial_angles or [],
            "max_sections": self.max_sections,
        }
        raw = await self.llm.chat_json(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        propositions = self._parse_propositions(raw.get("propositions") or raw.get("new_propositions") or [])
        script = self._parse_script(
            raw.get("script") if isinstance(raw.get("script"), dict) else raw,
            research_question,
            version=1,
        )
        return propositions, script

    async def update_script(
        self,
        research_question: str,
        propositions: list[Proposition],
        evidence: list,
        previous_script: InterviewScript,
        metrics: dict[str, Any],
    ) -> InterviewScript:
        evidence_briefing = self._build_evidence_briefing(propositions=propositions, evidence=evidence)
        payload = {
            "task": "Update interview script based on current state",
            "research_question": research_question,
            "propositions": [p.model_dump(mode="json") for p in propositions],
            "evidence_briefing": evidence_briefing,
            "previous_script": previous_script.model_dump(mode="json"),
            "metrics": metrics,
            "max_sections": self.max_sections,
        }
        raw = await self.llm.chat_json(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        version = previous_script.version + 1
        return self._parse_script(
            raw.get("script") if isinstance(raw.get("script"), dict) else raw,
            research_question,
            version=version,
        )

    async def generate_minimal_script(
        self,
        research_question: str,
        propositions: list[Proposition],
        metrics: dict[str, Any],
        version: int,
    ) -> InterviewScript:
        sections = [
            ScriptSection(
                proposition_id=p.id,
                priority="high" if idx == 0 else "medium",
                instruction="EXPLORE",
                main_question=f"Could you tell me more about {p.factor.lower()}?",
                probes=["Can you give a concrete example?", "What happened next?"],
                context="MVP fallback section",
            )
            for idx, p in enumerate(propositions[: self.max_sections])
        ]

        return InterviewScript(
            version=version,
            research_question=research_question,
            opening_question="Could you share your overall experience so far?",
            sections=sections,
            closing_question="What surprised you most about this experience?",
            wildcard="Is there anything important I have not asked about?",
            mode=metrics.get("mode", "divergent"),
            convergence_score=float(metrics.get("convergence_score", 0.0)),
            novelty_rate=float(metrics.get("novelty_rate", 1.0)),
            changes_summary="MVP fallback script generated",
        )

    def build_interviewer_prompt(self, script: InterviewScript) -> str:
        topic_blocks = []
        probe_lines = []
        for section in script.sections[: self.max_sections]:
            topic_blocks.append(
                "\n".join(
                    [
                        f"### Topic [{section.instruction}, priority: {section.priority.upper()}]",
                        f"- Main question: \"{section.main_question}\"",
                        f"- Probes: {' / '.join(section.probes)}",
                        f"- Context: {section.context}",
                    ]
                )
            )
            probe_lines.append(
                f"- {section.proposition_id}: {section.instruction} ({section.priority})"
            )

        rendered = self.interviewer_base_prompt
        rendered = rendered.replace("{opening_question}", script.opening_question)
        rendered = rendered.replace(
            "{propositions_and_questions}", "\n\n".join(topic_blocks) if topic_blocks else "No active topics"
        )
        rendered = rendered.replace(
            "{probe_instructions}", "\n".join(probe_lines) if probe_lines else "- Explore emerging themes"
        )
        rendered = rendered.replace("{closing_question}", script.closing_question)
        rendered = rendered.replace("{wildcard_question}", script.wildcard)
        return rendered

    def _build_evidence_briefing(self, propositions: list[Proposition], evidence: list) -> dict[str, Any]:
        interview_ids = sorted(
            {
                str(getattr(item, "interview_id", "")).strip()
                for item in evidence
                if str(getattr(item, "interview_id", "")).strip()
            }
        )
        proposition_brief = []
        mapped_evidence_ids: set[str] = set()

        for proposition in propositions:
            support_ids = [x for x in proposition.supporting_evidence if x]
            contradict_ids = [x for x in proposition.contradicting_evidence if x]
            mapped_evidence_ids.update(support_ids)
            mapped_evidence_ids.update(contradict_ids)
            proposition_brief.append(
                {
                    "id": proposition.id,
                    "factor": proposition.factor,
                    "mechanism": proposition.mechanism,
                    "outcome": proposition.outcome,
                    "status": proposition.status,
                    "confidence": proposition.confidence,
                    "support_count": len(support_ids),
                    "contradict_count": len(contradict_ids),
                }
            )

        unassigned_count = 0
        total_evidence = 0
        for item in evidence:
            evidence_id = str(getattr(item, "id", "")).strip()
            if not evidence_id:
                continue
            total_evidence += 1
            if evidence_id not in mapped_evidence_ids:
                unassigned_count += 1

        return {
            "total_evidence": total_evidence,
            "interviews_count": len(interview_ids),
            "unassigned_evidence_count": unassigned_count,
            "proposition_coverage": proposition_brief,
            "note": "Briefing is aggregate only; no respondent-specific quotes or personal references.",
        }

    def _parse_propositions(self, items: list[dict[str, Any]]) -> list[Proposition]:
        propositions: list[Proposition] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            factor = str(item.get("factor", "")).strip()
            mechanism = str(item.get("mechanism", "")).strip()
            outcome = str(item.get("outcome", "")).strip()
            if not (factor and mechanism and outcome):
                continue
            status = str(item.get("status", "untested")).lower()
            if status not in _VALID_PROPOSITION_STATUSES:
                status = "untested"
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            propositions.append(
                Proposition(
                    id=str(item.get("id", "")).strip(),
                    factor=factor,
                    mechanism=mechanism,
                    outcome=outcome,
                    confidence=max(0.0, min(1.0, confidence)),
                    status=status,
                )
            )
        return propositions

    def _parse_script(
        self,
        payload: dict[str, Any],
        research_question: str,
        version: int,
    ) -> InterviewScript:
        sections: list[ScriptSection] = []
        for item in payload.get("sections", []):
            if not isinstance(item, dict):
                continue
            proposition_id = str(item.get("proposition_id", "")).strip() or "P000"
            priority = str(item.get("priority", "medium")).lower()
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            instruction = str(item.get("instruction", "EXPLORE")).upper()
            if instruction not in {"EXPLORE", "VERIFY", "CHALLENGE", "SATURATED"}:
                instruction = "EXPLORE"
            main_question = str(item.get("main_question", "Could you tell me more?")).strip()
            probes = item.get("probes", [])
            if not isinstance(probes, list):
                probes = []
            sections.append(
                ScriptSection(
                    proposition_id=proposition_id,
                    priority=priority,
                    instruction=instruction,
                    main_question=main_question,
                    probes=[str(p) for p in probes if str(p).strip()][:3],
                    context=str(item.get("context", "")).strip(),
                )
            )

        return InterviewScript(
            version=version,
            generated_after_interview=payload.get("generated_after_interview"),
            research_question=research_question,
            opening_question=str(
                payload.get(
                    "opening_question",
                    "Could you share your experience with the hackathon so far?",
                )
            ),
            sections=sections[: self.max_sections],
            closing_question=str(
                payload.get("closing_question", "What surprised you most about this experience?")
            ),
            wildcard=str(
                payload.get("wildcard", "Is there anything important I have not asked about?")
            ),
            mode=(
                str(payload.get("mode", "divergent")).lower()
                if str(payload.get("mode", "divergent")).lower() in {"divergent", "convergent"}
                else "divergent"
            ),
            convergence_score=float(payload.get("convergence_score", 0.0)),
            novelty_rate=float(payload.get("novelty_rate", 1.0)),
            changes_summary=str(payload.get("changes_summary", "Script updated")),
        )
