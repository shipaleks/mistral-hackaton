from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from agents.llm_client import LLMClient
from agents.prompt_loader import load_prompt
from models.evidence import Evidence
from models.project import ProjectState


class SynthesizerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.system_prompt = load_prompt("synthesizer_system.txt")

    async def synthesize(self, project: ProjectState) -> str:
        result = await self.synthesize_with_meta(project)
        return result["report"]

    async def synthesize_with_meta(self, project: ProjectState) -> dict[str, Any]:
        if len(project.interview_store) == 0 or len(project.evidence_store) == 0:
            return {
                "report": self._no_data_report(project),
                "is_fallback": True,
                "fallback_reason": "No interview evidence in project",
            }

        quote_translations = await self.translate_evidence_quotes(project.evidence_store)
        evidence_payload = []
        for evidence in project.evidence_store:
            item = evidence.model_dump(mode="json")
            quote_en = str(quote_translations.get(evidence.id, "")).strip()
            if not quote_en and str(evidence.language or "").lower().startswith("en"):
                quote_en = evidence.quote.strip()
            if not quote_en:
                quote_en = "Translation unavailable"
            item["quote_english"] = quote_en
            evidence_payload.append(item)

        payload = {
            "research_question": project.research_question,
            "evidence": evidence_payload,
            "propositions": [p.model_dump(mode="json") for p in project.proposition_store],
            "metrics": project.metrics.model_dump(mode="json"),
            "interviews": len(project.interview_store),
            "script_versions": len(project.script_versions),
            "grounding_rules": {
                "must_use_only_provided_evidence": True,
                "must_not_invent_quotes": True,
                "must_not_invent_participant_labels": True,
                "translated_quotes_must_include_original_marker": '[original: "..."]',
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
            print(f"[synthesizer] llm generation failed: {err}")

        if llm_report and self._is_grounded(llm_report, project) and not self._has_non_english_quotes(
            llm_report
        ):
            return {
                "report": llm_report,
                "is_fallback": False,
                "fallback_reason": None,
            }

        reason = self._format_fallback_reason(llm_error)
        return {
            "report": self._grounded_fallback_report(project, reason, quote_translations),
            "is_fallback": True,
            "fallback_reason": reason,
        }

    def _is_grounded(self, report: str, project: ProjectState) -> bool:
        text = report or ""
        if re.search(r"\bParticipant\s+[A-Z]\b", text):
            return False

        evidence_quotes = [self._norm(e.quote) for e in project.evidence_store if e.quote.strip()]
        if not evidence_quotes:
            return False

        original_markers = self._extract_original_markers(text)
        if original_markers:
            for original in original_markers:
                if not self._matches_any_evidence_quote(original, evidence_quotes):
                    return False
            return True

        quote_candidates = re.findall(r"[\"“](.*?)[\"”]", text, flags=re.DOTALL)
        for candidate in quote_candidates:
            norm_candidate = self._norm(candidate)
            if len(norm_candidate) < 10:
                continue
            if not self._matches_any_evidence_quote(norm_candidate, evidence_quotes):
                return False

        return True

    def _matches_any_evidence_quote(
        self,
        quote_or_norm: str,
        evidence_quotes_norm: list[str],
    ) -> bool:
        norm_candidate = self._norm(quote_or_norm)
        if not norm_candidate:
            return False
        return any(
            norm_candidate in evidence_quote or evidence_quote in norm_candidate
            for evidence_quote in evidence_quotes_norm
        )

    def _extract_original_markers(self, report: str) -> list[str]:
        matches = re.findall(r"\[original:\s*\"(.*?)\"\s*\]", report, flags=re.IGNORECASE | re.DOTALL)
        matches.extend(
            re.findall(r"\[original:\s*'(.*?)'\s*\]", report, flags=re.IGNORECASE | re.DOTALL)
        )
        return [m.strip() for m in matches if m and m.strip()]

    def _has_non_english_quotes(self, report: str) -> bool:
        cleaned = re.sub(
            r"\[original:\s*(\".*?\"|'.*?')\s*\]",
            "",
            report,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Reject clearly non-English scripts outside explicit original markers.
        return bool(re.search(r"[А-Яа-яЁё\u3040-\u30ff\u3400-\u9fff]", cleaned))

    def _format_fallback_reason(self, llm_error: Exception | None) -> str:
        if llm_error is None:
            return "LLM output was not grounded/safe"

        detail = str(llm_error).strip() or llm_error.__class__.__name__
        detail = re.sub(r"\s+", " ", detail)
        if len(detail) > 260:
            detail = f"{detail[:257]}..."
        return f"LLM generation failed: {detail}"

    async def translate_evidence_quotes(self, evidence_items: list[Evidence]) -> dict[str, str]:
        translations: dict[str, str] = {}
        source_quotes: list[dict[str, str]] = []

        for evidence in evidence_items:
            quote = evidence.quote.strip()
            if not quote:
                continue

            language = (evidence.language or "").strip().lower()
            if language.startswith("en"):
                translations[evidence.id] = quote
                continue

            source_quotes.append(
                {
                    "id": evidence.id,
                    "language": language or "unknown",
                    "quote": quote,
                }
            )

        if not source_quotes or not hasattr(self.llm, "chat_json"):
            return translations

        system_prompt = (
            "You translate interview quotes into English. "
            "Return strict JSON with key `translations` containing "
            "a list of {id, english}. Keep meaning faithful. No extra commentary."
        )

        try:
            result = await self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"quotes": source_quotes}, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
        except Exception:
            return translations

        items: list[dict] = []
        if isinstance(result.get("translations"), list):
            items = [x for x in result.get("translations", []) if isinstance(x, dict)]
        elif isinstance(result.get("items"), list):
            items = [x for x in result.get("items", []) if isinstance(x, dict)]
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, str):
                    items.append({"id": key, "english": value})

        for item in items:
            evidence_id = str(item.get("id") or "").strip()
            english = str(
                item.get("english")
                or item.get("translation")
                or item.get("translated")
                or ""
            ).strip()
            if evidence_id and english:
                translations[evidence_id] = english

        return translations

    def _grounded_fallback_report(
        self,
        project: ProjectState,
        reason: str,
        quote_translations: dict[str, str],
    ) -> str:
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
                        original_quote = self._escape_inline_quote(ev.quote)
                        translated = str(quote_translations.get(ev.id, "")).strip()
                        if translated:
                            english_quote = self._escape_inline_quote(translated)
                        elif str(ev.language or "").lower().startswith("en"):
                            english_quote = original_quote
                        else:
                            english_quote = "Translation unavailable"
                        if self._norm(original_quote) == self._norm(english_quote):
                            quote_lines.append(f"- [{ev.id}] \"{english_quote}\"")
                        else:
                            quote_lines.append(
                                f"- [{ev.id}] \"{english_quote}\" [original: \"{original_quote}\"]"
                            )

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
            original_quote = self._escape_inline_quote(ev.quote)
            translated = str(quote_translations.get(ev.id, "")).strip()
            if translated:
                english_quote = self._escape_inline_quote(translated)
            elif str(ev.language or "").lower().startswith("en"):
                english_quote = original_quote
            else:
                english_quote = "Translation unavailable"
            if self._norm(original_quote) == self._norm(english_quote):
                lines.append(f"  Quote (EN): \"{english_quote}\"")
            else:
                lines.append(
                    f"  Quote (EN): \"{english_quote}\" [original: \"{original_quote}\"]"
                )
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
        normalized = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).replace("_", " ")
        return re.sub(r"\s+", " ", normalized).strip()

    def _escape_inline_quote(self, text: str) -> str:
        return str(text or "").strip().replace('"', "'")
