from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from agents.llm_client import LLMClient
from agents.prompt_loader import load_prompt
from models.project import ProjectState


class SynthesizerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("synthesizer_system.txt")

    async def synthesize(self, project: ProjectState) -> str:
        if len(project.interview_store) == 0 or len(project.evidence_store) == 0:
            return self._no_data_report(project)

        payload = {
            "research_question": project.research_question,
            "evidence": [e.model_dump(mode="json") for e in project.evidence_store],
            "propositions": [p.model_dump(mode="json") for p in project.proposition_store],
            "metrics": project.metrics.model_dump(mode="json"),
            "interviews": len(project.interview_store),
            "script_versions": len(project.script_versions),
            "grounding_rules": {
                "must_use_only_provided_evidence": True,
                "must_not_invent_quotes": True,
                "must_not_invent_participant_labels": True,
            },
        }

        llm_report: str | None = None
        llm_error: Exception | None = None
        try:
            llm_report = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as err:
            llm_error = err

        if llm_report and self._is_grounded(llm_report, project):
            return llm_report

        reason = "LLM generation failed" if llm_error else "LLM output was not grounded in evidence"
        return self._grounded_fallback_report(project, reason)

    def _is_grounded(self, report: str, project: ProjectState) -> bool:
        text = report or ""
        if re.search(r"\bParticipant\s+[A-Z]\b", text):
            return False

        evidence_quotes = [self._norm(e.quote) for e in project.evidence_store if e.quote.strip()]
        if not evidence_quotes:
            return False

        quote_candidates = re.findall(r"[\"“](.*?)[\"”]", text, flags=re.DOTALL)
        for candidate in quote_candidates:
            norm_candidate = self._norm(candidate)
            if len(norm_candidate) < 16:
                continue
            matched = any(
                norm_candidate in evidence_quote or evidence_quote in norm_candidate
                for evidence_quote in evidence_quotes
            )
            if not matched:
                return False

        return True

    def _grounded_fallback_report(self, project: ProjectState, reason: str) -> str:
        evidence_by_id = {e.id: e for e in project.evidence_store}
        total_interviews = len(project.interview_store)
        total_evidence = len(project.evidence_store)

        lines: list[str] = []
        lines.append("# Qualitative Research Report (Grounded Fallback)")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append(
            f"This report is generated from stored project evidence only. "
            f"Data coverage: {total_interviews} interview(s), {total_evidence} evidence item(s)."
        )
        lines.append(
            f"Current convergence: {project.metrics.convergence_score:.2f}, "
            f"novelty: {project.metrics.novelty_rate:.2f}, mode: {project.metrics.mode}."
        )
        lines.append("")

        lines.append("## Methodology Note")
        lines.append("Interviews were processed through the webhook -> analyst -> designer pipeline.")
        lines.append("Only stored evidence/proposition objects were used in this report.")
        lines.append(f"Fallback reason: {reason}.")
        lines.append("")

        lines.append("## Core Findings")
        propositions = sorted(
            project.proposition_store,
            key=lambda p: (p.confidence, len(p.supporting_evidence)),
            reverse=True,
        )

        if not propositions:
            lines.append("No propositions available yet.")
        else:
            for idx, prop in enumerate(propositions[:5], start=1):
                lines.append(
                    f"### {idx}. {prop.factor} -> {prop.mechanism} -> {prop.outcome}"
                )
                lines.append(
                    f"Status: `{prop.status}` | Confidence: {prop.confidence:.2f} | "
                    f"Supporting: {len(prop.supporting_evidence)} | Contradicting: {len(prop.contradicting_evidence)}"
                )

                quote_lines = []
                for eid in prop.supporting_evidence[:3]:
                    ev = evidence_by_id.get(eid)
                    if ev and ev.quote.strip():
                        quote_lines.append(f"- [{ev.id}] \"{ev.quote.strip()}\"")

                if quote_lines:
                    lines.append("Supporting quotes:")
                    lines.extend(quote_lines)
                else:
                    lines.append("Supporting quotes: none linked yet.")
                lines.append("")

        lines.append("## Contradictions & Caveats")
        contradicted = [p for p in project.proposition_store if len(p.contradicting_evidence) > 0]
        if contradicted:
            for prop in contradicted[:5]:
                lines.append(
                    f"- `{prop.id}` has {len(prop.contradicting_evidence)} contradicting evidence item(s)."
                )
        else:
            lines.append("- No explicit contradicting mappings were recorded yet.")
        lines.append("")

        lines.append("## Raw Evidence Samples")
        for ev in project.evidence_store[:8]:
            lines.append(
                f"- [{ev.id}] ({ev.language}) {ev.factor} -> {ev.mechanism} -> {ev.outcome}"
            )
            lines.append(f"  Quote: \"{ev.quote.strip()}\"")
        lines.append("")

        lines.append("## Next Steps")
        lines.append("1. Collect additional interviews to improve coverage.")
        lines.append("2. Re-run report generation after new evidence is ingested.")
        lines.append("3. Manually inspect proposition mappings with low confidence.")
        lines.append("")
        lines.append(
            f"_Generated at {datetime.now(timezone.utc).isoformat()} from in-project data only._"
        )

        return "\n".join(lines)

    def _no_data_report(self, project: ProjectState) -> str:
        return "\n".join(
            [
                "# Report Not Ready",
                "",
                "## Why report generation is blocked",
                "No interview evidence is currently stored for this project.",
                "",
                "## Project state",
                f"- Project ID: `{project.id}`",
                f"- Research question: {project.research_question}",
                f"- Interviews: {len(project.interview_store)}",
                f"- Evidence items: {len(project.evidence_store)}",
                "",
                "## What to do next",
                "1. Verify ElevenLabs webhook URL points to `/api/webhook/elevenlabs`.",
                "2. Complete at least one interview and confirm participants/evidence counters increase.",
                "3. Run `Finish & Generate Report` again.",
            ]
        )

    def _norm(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9а-яё\s]", "", text.lower())).strip()
