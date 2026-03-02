from __future__ import annotations

from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection
from services.script_safety import ScriptSafetyGuard


def _script_with_personal_memory() -> InterviewScript:
    return InterviewScript(
        version=2,
        research_question="What shaped your onboarding experience in the first month?",
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
        research_question="What shaped your onboarding experience in the first month?",
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
        research_question="What shaped your onboarding experience in the first month?",
        opening_question="How was your onboarding experience in the first month?",
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
        research_question="What shaped your onboarding experience in the first month?",
        propositions=propositions,
    )

    assert result.topic_redirect_applied
    assert "research question" in result.script.sections[0].main_question.lower()


def _script_with_russian_personal_memory() -> InterviewScript:
    return InterviewScript(
        version=2,
        research_question="Какие факторы влияют на ваш опыт адаптации?",
        opening_question="Ранее вы упоминали некоторые трудности. Можем продолжить?",
        sections=[
            ScriptSection(
                proposition_id="P001",
                priority="high",
                instruction="CHALLENGE",
                main_question="Вы говорили, что работали в одиночку. Как это ощущалось?",
                probes=[
                    "Вы рассказывали, что правила команды были неясными. Расскажите подробнее?",
                    "Как мы обсуждали, что случилось дальше?",
                ],
                context="Ранее вы упоминали выгорание и конфликт в команде.",
            )
        ],
        closing_question="Из того что вы сказали, что было самым сложным?",
        wildcard="Есть ещё что-то, о чём вы мне говорили?",
        mode="divergent",
        convergence_score=0.2,
        novelty_rate=0.7,
        changes_summary="Updated from previous respondent",
    )


def test_script_safety_guard_detects_russian_personal_refs() -> None:
    guard = ScriptSafetyGuard()
    script = _script_with_russian_personal_memory()
    propositions = [
        Proposition(
            id="P001",
            factor="Динамика формирования команды",
            mechanism="Ограничения правил",
            outcome="Качество сотрудничества",
            confidence=0.4,
            status="exploring",
        )
    ]

    result = guard.enforce(
        script=script,
        research_question="Какие факторы влияют на ваш опыт адаптации?",
        propositions=propositions,
        language="ru",
    )

    assert result.status in {"sanitized", "fallback"}
    assert result.violations_count > 0
    assert "вы упоминали" not in result.script.opening_question.lower()


def test_script_safety_guard_russian_topic_redirect() -> None:
    guard = ScriptSafetyGuard()
    script = InterviewScript(
        version=1,
        research_question="Какие факторы влияют на ваш опыт адаптации?",
        opening_question="Расскажите о вашем опыте адаптации.",
        sections=[
            ScriptSection(
                proposition_id="P001",
                priority="high",
                instruction="EXPLORE",
                main_question="Расскажите о вашем проекте и стеке технологий.",
                probes=["Какую кодовую базу вы использовали?"],
                context="Техническая реализация",
            )
        ],
        closing_question="Что ещё?",
        wildcard="Дополнительные заметки?",
        mode="divergent",
        convergence_score=0.0,
        novelty_rate=1.0,
        changes_summary="Init",
    )
    propositions = [
        Proposition(
            id="P001",
            factor="Ограничения проекта",
            mechanism="Давление сроков",
            outcome="Качество результата",
            confidence=0.2,
            status="exploring",
        )
    ]

    result = guard.enforce(
        script=script,
        research_question="Какие факторы влияют на ваш опыт адаптации?",
        propositions=propositions,
        language="ru",
    )

    assert result.topic_redirect_applied
