from __future__ import annotations

import json
from typing import Any

from agents.llm_client import LLMClient
from agents.prompt_loader import load_prompt
from models.analysis import AnalysisMetrics, AnalysisResult, EvidenceMapping, PropositionUpdate
from models.evidence import Evidence
from models.proposition import Proposition


_VALID_STATUSES = {
    "untested",
    "exploring",
    "confirmed",
    "challenged",
    "saturated",
    "weak",
    "merged",
}


class AnalystAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("analyst_system.txt")

    async def analyze_interview(
        self,
        transcript: str,
        existing_evidence: list[Evidence],
        existing_propositions: list[Proposition],
        interview_id: str,
        interview_index: int,
    ) -> AnalysisResult:
        user_payload = {
            "task": "Analyze a single interview and return JSON only",
            "interview_id": interview_id,
            "transcript": transcript,
            "existing_evidence": [e.model_dump(mode="json") for e in existing_evidence],
            "existing_propositions": [p.model_dump(mode="json") for p in existing_propositions],
        }

        payload = await self.llm.chat_json(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.3,
            max_tokens=8192,
        )

        return self._coerce_analysis_result(payload, interview_id, interview_index)

    def _coerce_analysis_result(
        self, payload: dict[str, Any], interview_id: str, interview_index: int
    ) -> AnalysisResult:
        new_evidence: list[Evidence] = []
        for item in payload.get("new_evidence", []):
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote", "")).strip()
            quote_english = str(item.get("quote_english", "")).strip()
            interpretation = str(item.get("interpretation", "")).strip()
            factor = str(item.get("factor", "")).strip()
            mechanism = str(item.get("mechanism", "")).strip()
            outcome = str(item.get("outcome", "")).strip()
            if not (quote and interpretation and factor and mechanism and outcome):
                continue

            tags = item.get("tags")
            if not isinstance(tags, list):
                tags = []

            language = str(item.get("language", "en")).strip() or "en"
            language_lc = language.lower()
            if language_lc.startswith("en"):
                translation_status = "native_en"
                quote_english = quote
            elif quote_english:
                translation_status = "translated"
            else:
                translation_status = "pending"

            new_evidence.append(
                Evidence(
                    id=str(item.get("id", "")).strip(),
                    interview_id=interview_id,
                    quote=quote,
                    quote_english=quote_english or None,
                    translation_status=translation_status,
                    interpretation=interpretation,
                    factor=factor,
                    mechanism=mechanism,
                    outcome=outcome,
                    tags=[str(t) for t in tags if str(t).strip()],
                    language=language,
                )
            )

        evidence_mappings = self._coerce_mappings(payload.get("evidence_mappings", []))
        retroactive_mappings = self._coerce_mappings(payload.get("retroactive_mappings", []))

        new_propositions: list[Proposition] = []
        for item in payload.get("new_propositions", []):
            if not isinstance(item, dict):
                continue
            factor = str(item.get("factor", "")).strip()
            mechanism = str(item.get("mechanism", "")).strip()
            outcome = str(item.get("outcome", "")).strip()
            if not (factor and mechanism and outcome):
                continue

            status = str(item.get("status", "untested")).strip().lower()
            if status not in _VALID_STATUSES:
                status = "untested"
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                first_seen = int(item.get("first_seen_interview", interview_index))
            except (TypeError, ValueError):
                first_seen = interview_index
            try:
                last_updated = int(item.get("last_updated_interview", interview_index))
            except (TypeError, ValueError):
                last_updated = interview_index

            new_propositions.append(
                Proposition(
                    id=str(item.get("id", "")).strip(),
                    factor=factor,
                    mechanism=mechanism,
                    outcome=outcome,
                    confidence=max(0.0, min(1.0, confidence)),
                    status=status,
                    supporting_evidence=[str(x) for x in item.get("supporting_evidence", []) if str(x)],
                    contradicting_evidence=[
                        str(x) for x in item.get("contradicting_evidence", []) if str(x)
                    ],
                    first_seen_interview=first_seen,
                    last_updated_interview=last_updated,
                )
            )

        proposition_updates: list[PropositionUpdate] = []
        for item in payload.get("proposition_updates", []):
            if not isinstance(item, dict):
                continue
            prop_id = str(item.get("id", "")).strip()
            if not prop_id:
                continue
            status = str(item.get("new_status", "exploring")).strip().lower()
            if status not in _VALID_STATUSES:
                status = "exploring"
            try:
                confidence = float(item.get("new_confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            proposition_updates.append(
                PropositionUpdate(
                    id=prop_id,
                    new_confidence=max(0.0, min(1.0, confidence)),
                    new_status=status,
                )
            )

        metrics_payload = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        try:
            convergence_score = float(metrics_payload.get("convergence_score", 0.0))
        except (TypeError, ValueError):
            convergence_score = 0.0
        try:
            novelty_rate = float(metrics_payload.get("novelty_rate", 1.0))
        except (TypeError, ValueError):
            novelty_rate = 1.0
        mode = str(metrics_payload.get("mode", "divergent")).strip().lower()
        if mode not in {"divergent", "convergent"}:
            mode = "divergent"

        result = AnalysisResult(
            new_evidence=new_evidence,
            evidence_mappings=evidence_mappings,
            new_propositions=new_propositions,
            retroactive_mappings=retroactive_mappings,
            proposition_updates=proposition_updates,
            prunes=[str(x) for x in payload.get("prunes", []) if str(x)],
            metrics=AnalysisMetrics(
                convergence_score=max(0.0, min(1.0, convergence_score)),
                novelty_rate=max(0.0, min(1.0, novelty_rate)),
                mode=mode,
            ),
        )
        return result

    def _coerce_mappings(self, items: list[Any]) -> list[EvidenceMapping]:
        mappings: list[EvidenceMapping] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("relationship", "")).strip().lower()
            if rel not in {"supports", "contradicts"}:
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            proposition_id = str(item.get("proposition_id", "")).strip()
            if not evidence_id or not proposition_id:
                continue
            mappings.append(
                EvidenceMapping(
                    evidence_id=evidence_id,
                    proposition_id=proposition_id,
                    relationship=rel,
                )
            )
        return mappings
