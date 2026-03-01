from __future__ import annotations

from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection
from services.script_safety import ScriptSafetyGuard


def _script_with_personal_memory() -> InterviewScript:
    return InterviewScript(
        version=2,
        research_question="What is your experience with this hackathon so far?",
        opening_question="Earlier you mentioned some difficulties. Can we continue?",
        sections=[
            ScriptSection(
                proposition_id="P001",
                priority="high",
                instruction="CHALLENGE",
                main_question="Earlier, you mentioned working alone. How did that feel?",
                probes=[
                    "You said the team rules were unclear. Can you explain?",
                    "As we discussed, what happened next in your project implementation?",
                ],
                context="Earlier you mentioned burnout and team conflict in detail.",
            )
        ],
        closing_question="From what you said, what was hardest?",
        wildcard="Anything else you told me before?",
        mode="divergent",
        convergence_score=0.2,
        novelty_rate=0.7,
        changes_summary="Updated from previous respondent",
    )


def test_script_safety_guard_detects_and_sanitizes() -> None:
    guard = ScriptSafetyGuard()
    script = _script_with_personal_memory()
    propositions = [
        Proposition(
            id="P001",
            factor="Team formation dynamics",
            mechanism="Rule constraints",
            outcome="Collaboration quality",
            confidence=0.4,
            status="exploring",
        )
    ]

    result = guard.enforce(
        script=script,
        research_question="What is your experience with this hackathon so far?",
        propositions=propositions,
    )

    assert result.status in {"sanitized", "fallback"}
    assert result.violations_count > 0
    assert "you mentioned" not in result.script.opening_question.lower()
    assert "you mentioned" not in result.script.sections[0].main_question.lower()
    assert "you said" not in result.script.sections[0].probes[0].lower()


def test_script_safety_guard_triggers_topic_redirect() -> None:
    guard = ScriptSafetyGuard()
    script = InterviewScript(
        version=1,
        research_question="What is your experience with this hackathon so far?",
        opening_question="How is your experience with this hackathon so far?",
        sections=[
            ScriptSection(
                proposition_id="P001",
                priority="high",
                instruction="EXPLORE",
                main_question="Tell me about your project implementation and tech stack decisions.",
                probes=["What frameworks did you choose for your codebase?"],
                context="Technical implementation details",
            )
        ],
        closing_question="Anything else?",
        wildcard="Any final notes?",
        mode="divergent",
        convergence_score=0.0,
        novelty_rate=1.0,
        changes_summary="Init",
    )
    propositions = [
        Proposition(
            id="P001",
            factor="Project constraints",
            mechanism="Time pressure",
            outcome="Delivery quality",
            confidence=0.2,
            status="exploring",
        )
    ]

    result = guard.enforce(
        script=script,
        research_question="What is your experience with this hackathon so far?",
        propositions=propositions,
    )

    assert result.topic_redirect_applied
    assert "hackathon" in result.script.sections[0].main_question.lower()

