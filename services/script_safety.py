from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection


_PERSONAL_PATTERNS = [
    re.compile(r"\bearlier\s+you\s+mentioned\b", re.IGNORECASE),
    re.compile(r"\byou\s+(said|told|described|shared|mentioned)\b", re.IGNORECASE),
    re.compile(r"\bas\s+we\s+discussed\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+what\s+you\s+said\b", re.IGNORECASE),
]

_TOPIC_DRIFT_PATTERNS = [
    re.compile(r"\byour\s+project\b", re.IGNORECASE),
    re.compile(r"\btech\s+stack\b", re.IGNORECASE),
    re.compile(r"\bcodebase\b", re.IGNORECASE),
    re.compile(r"\bimplementation\b", re.IGNORECASE),
    re.compile(r"\bapi\s+integration\b", re.IGNORECASE),
    re.compile(r"\binfrastructure\b", re.IGNORECASE),
]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w]+", str(text or "").lower()) if len(token) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return intersection / union


@dataclass
class ScriptViolation:
    section_index: int | None
    field: str
    reason: str
    value: str


@dataclass
class ScriptSafetyResult:
    script: InterviewScript
    status: str
    violations: list[ScriptViolation] = field(default_factory=list)
    topic_redirect_applied: bool = False

    @property
    def violations_count(self) -> int:
        return len(self.violations)


class ScriptSafetyGuard:
    def validate_script(self, script: InterviewScript) -> list[ScriptViolation]:
        violations: list[ScriptViolation] = []

        self._check_text(
            text=script.opening_question,
            field="opening_question",
            section_index=None,
            violations=violations,
        )
        self._check_text(
            text=script.closing_question,
            field="closing_question",
            section_index=None,
            violations=violations,
        )
        self._check_text(
            text=script.wildcard,
            field="wildcard",
            section_index=None,
            violations=violations,
        )

        for idx, section in enumerate(script.sections):
            self._check_text(
                text=section.main_question,
                field="main_question",
                section_index=idx,
                violations=violations,
            )
            self._check_text(
                text=section.context,
                field="context",
                section_index=idx,
                violations=violations,
            )
            for probe_idx, probe in enumerate(section.probes):
                self._check_text(
                    text=probe,
                    field=f"probes[{probe_idx}]",
                    section_index=idx,
                    violations=violations,
                )
        return violations

    def enforce(
        self,
        script: InterviewScript,
        research_question: str,
        propositions: list[Proposition],
    ) -> ScriptSafetyResult:
        violations = self.validate_script(script)
        proposition_index = {p.id: p for p in propositions}
        topic_redirect_applied = False
        safe_sections: list[ScriptSection] = []
        script_changed = False

        for section in script.sections:
            proposition = proposition_index.get(section.proposition_id)
            original_main = str(section.main_question or "").strip()
            main_question = self._sanitize_text(section.main_question)
            if self._has_personal_reference(main_question) or not main_question.strip():
                main_question = self._fallback_question(proposition, research_question)

            if self._is_topic_drift(main_question, research_question):
                main_question = self._topic_redirect_question(main_question, research_question)
                topic_redirect_applied = True

            probes: list[str] = []
            for probe in section.probes[:3]:
                cleaned = self._sanitize_text(probe)
                if not cleaned:
                    continue
                if self._is_topic_drift(cleaned, research_question):
                    cleaned = self._topic_redirect_probe(cleaned, research_question)
                    topic_redirect_applied = True
                if cleaned not in probes:
                    probes.append(cleaned)

            if not probes:
                probes = self._default_probes(research_question)

            context = self._safe_context(section.proposition_id, proposition)
            if (
                main_question.strip() != original_main
                or context.strip() != str(section.context or "").strip()
                or probes != list(section.probes[:3])
            ):
                script_changed = True

            safe_sections.append(
                ScriptSection(
                    proposition_id=section.proposition_id,
                    priority=section.priority,
                    instruction=section.instruction,
                    main_question=main_question,
                    probes=probes[:3],
                    context=context,
                )
            )

        safe_opening = self._sanitize_text(script.opening_question)
        if self._has_personal_reference(safe_opening) or not safe_opening.strip():
            safe_opening = self._default_opening(research_question)

        safe_closing = self._sanitize_text(script.closing_question)
        if self._has_personal_reference(safe_closing) or not safe_closing.strip():
            safe_closing = self._default_closing(research_question)

        safe_wildcard = self._sanitize_text(script.wildcard)
        if self._has_personal_reference(safe_wildcard) or not safe_wildcard.strip():
            safe_wildcard = (
                "Is there anything else about your experience with this research topic that we should capture?"
            )

        safe_script = script.model_copy(deep=True)
        safe_script.opening_question = safe_opening
        safe_script.sections = safe_sections
        safe_script.closing_question = safe_closing
        safe_script.wildcard = safe_wildcard

        if (
            safe_opening.strip() != str(script.opening_question or "").strip()
            or safe_closing.strip() != str(script.closing_question or "").strip()
            or safe_wildcard.strip() != str(script.wildcard or "").strip()
        ):
            script_changed = True

        status = "ok"
        if violations:
            status = "sanitized" if safe_sections else "fallback"
        if not safe_sections:
            safe_script.sections = [
                ScriptSection(
                    proposition_id="P000",
                    priority="high",
                    instruction="EXPLORE",
                    main_question=self._default_opening(research_question),
                    probes=self._default_probes(research_question),
                    context="Fallback section generated by safety guard",
                )
            ]
            script_changed = True

        if not script_changed and not violations and not topic_redirect_applied:
            return ScriptSafetyResult(script=script, status="ok", violations=[])

        return ScriptSafetyResult(
            script=safe_script,
            status=status,
            violations=violations,
            topic_redirect_applied=topic_redirect_applied,
        )

    def _check_text(
        self,
        text: str,
        field: str,
        section_index: int | None,
        violations: list[ScriptViolation],
    ) -> None:
        value = str(text or "").strip()
        if not value:
            return
        for pattern in _PERSONAL_PATTERNS:
            if pattern.search(value):
                violations.append(
                    ScriptViolation(
                        section_index=section_index,
                        field=field,
                        reason="personal_reference",
                        value=value,
                    )
                )
                break

    def _sanitize_text(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""

        replacements = [
            (r"\b[Ee]arlier,\s*you\s+mentioned\b", "Some participants mentioned"),
            (r"\b[Ee]arlier\s+you\s+mentioned\b", "Some participants mentioned"),
            (r"\b[Yy]ou\s+(said|told|described|shared|mentioned)\b", "Some participants reported"),
            (r"\b[Aa]s\s+we\s+discussed\b", "From previous interviews"),
        ]
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value)
        return re.sub(r"\s+", " ", value).strip()

    def _has_personal_reference(self, text: str) -> bool:
        value = str(text or "")
        return any(pattern.search(value) for pattern in _PERSONAL_PATTERNS)

    def _is_topic_drift(self, text: str, research_question: str) -> bool:
        value = str(text or "")
        rq_tokens = _tokenize(research_question or "")
        text_tokens = _tokenize(value)
        if rq_tokens and _jaccard(rq_tokens, text_tokens) >= 0.18:
            return False
        return any(pattern.search(value) for pattern in _TOPIC_DRIFT_PATTERNS)

    def _topic_redirect_question(self, text: str, research_question: str) -> str:
        return (
            f"Could you connect this back to the main research question: "
            f"'{research_question}'?"
        )

    def _topic_redirect_probe(self, text: str, research_question: str) -> str:
        return "How did this influence your experience with the core research topic?"

    def _fallback_question(
        self,
        proposition: Proposition | None,
        research_question: str,
    ) -> str:
        if proposition is None:
            return self._default_opening(research_question)
        return (
            f"How did {proposition.factor.lower()} influence your experience with this topic, "
            f"and what outcomes did it create?"
        )

    def _safe_context(self, proposition_id: str, proposition: Proposition | None) -> str:
        if proposition is None:
            return f"Explore proposition {proposition_id} in aggregate, without respondent-specific references."
        return (
            f"Aggregate focus for {proposition_id}: {proposition.factor} -> {proposition.mechanism} -> "
            f"{proposition.outcome}. Keep wording respondent-agnostic."
        )

    def _default_opening(self, research_question: str) -> str:
        return (
            f"Could you describe your experience related to this research question: "
            f"'{research_question}'?"
        )

    def _default_closing(self, research_question: str) -> str:
        return (
            "Before we end, what was the most important part of your experience related to this research question?"
        )

    def _default_probes(self, research_question: str) -> list[str]:
        return [
            "Can you give a concrete example related to this topic?",
            "What impact did this have on your experience?",
            "Did this change over time?",
        ]
