from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.analyst import AnalystAgent
from agents.designer import DesignerAgent
from models.analysis import AnalysisResult
from models.interview import Interview
from services.elevenlabs_service import ElevenLabsService
from services.project_service import ProjectService
from services.script_safety import ScriptSafetyGuard
from services.sse_manager import SSEManager
from services.visualization import apply_heuristic_links


class Pipeline:
    def __init__(
        self,
        project_service: ProjectService,
        analyst: AnalystAgent,
        designer: DesignerAgent,
        elevenlabs: ElevenLabsService,
        sse: SSEManager,
        script_safety: ScriptSafetyGuard | None = None,
    ) -> None:
        self.project_service = project_service
        self.analyst = analyst
        self.designer = designer
        self.elevenlabs = elevenlabs
        self.sse = sse
        self.script_safety = script_safety or ScriptSafetyGuard()

    async def process_interview(
        self,
        project_id: str,
        transcript: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = self.project_service.load_project(project_id)

        if conversation_id in project.processed_conversation_ids:
            return {"status": "duplicate", "conversation_id": conversation_id}

        if project.status == "draft":
            project.status = "running"

        report_became_stale = project.status == "done"

        interview = Interview(
            id=self.project_service.next_interview_id(project),
            conversation_id=conversation_id,
            transcript=transcript,
            metadata=metadata or {},
        )
        self.project_service.add_interview(project, interview)

        result = await self.analyst.analyze_interview(
            transcript=transcript,
            existing_evidence=project.evidence_store,
            existing_propositions=project.proposition_store,
            interview_id=interview.id,
            interview_index=len(project.interview_store),
        )
        self._apply_analysis_result(project, result)
        await self.emit_analysis_events(project_id, result)

        heuristic_changed, heuristic_added = apply_heuristic_links(project)

        previous_script = project.current_script
        try:
            if previous_script is None:
                new_script = await self.designer.generate_minimal_script(
                    research_question=project.research_question,
                    propositions=project.proposition_store,
                    metrics=project.metrics.model_dump(),
                    version=1,
                )
            else:
                new_script = await self.designer.update_script(
                    research_question=project.research_question,
                    propositions=project.proposition_store,
                    evidence=project.evidence_store,
                    previous_script=previous_script,
                    metrics=project.metrics.model_dump(),
                )
        except Exception:
            fallback_version = 1 if previous_script is None else previous_script.version + 1
            new_script = await self.designer.generate_minimal_script(
                research_question=project.research_question,
                propositions=project.proposition_store,
                metrics=project.metrics.model_dump(),
                version=fallback_version,
            )
            new_script.changes_summary = "Fallback script generated after designer failure"

        safety_result = self.script_safety.enforce(
            script=new_script,
            research_question=project.research_question,
            propositions=project.proposition_store,
        )
        new_script = safety_result.script
        project.prompt_safety_status = safety_result.status
        project.prompt_safety_violations_count = safety_result.violations_count

        if safety_result.status in {"sanitized", "fallback"}:
            marker = f"safety_guard={safety_result.status} violations={safety_result.violations_count}"
            if marker not in new_script.changes_summary:
                summary = new_script.changes_summary.strip() or "Script updated"
                new_script.changes_summary = f"{summary} [{marker}]"

        self.project_service.add_script(project, new_script)

        project.sync_pending = False
        project.sync_pending_script_version = None
        if project.elevenlabs_agent_id:
            full_prompt = self.designer.build_interviewer_prompt(new_script)
            try:
                await self.elevenlabs.update_agent_prompt(project.elevenlabs_agent_id, full_prompt)
                project.last_prompt_update_at = datetime.now(timezone.utc)
            except Exception:
                project.sync_pending = True
                project.sync_pending_script_version = new_script.version

        if report_became_stale:
            project.report_stale = True

        project.processed_conversation_ids.append(conversation_id)
        self.project_service.save_project(project)

        await self.sse.emit(
            project_id,
            "script_updated",
            {
                "version": new_script.version,
                "changes_summary": new_script.changes_summary,
                "sync_pending": project.sync_pending,
                "prompt_safety_status": project.prompt_safety_status,
                "prompt_safety_violations_count": project.prompt_safety_violations_count,
            },
        )

        if safety_result.status in {"sanitized", "fallback"}:
            await self.sse.emit(
                project_id,
                "prompt_sanitized",
                {
                    "project_id": project.id,
                    "script_version": new_script.version,
                    "status": safety_result.status,
                    "violations_count": safety_result.violations_count,
                },
            )

        if safety_result.topic_redirect_applied:
            await self.sse.emit(
                project_id,
                "topic_redirect_applied",
                {
                    "project_id": project.id,
                    "script_version": new_script.version,
                },
            )

        if report_became_stale:
            await self.sse.emit(
                project_id,
                "report_stale",
                {
                    "project_id": project.id,
                    "status": project.status,
                    "report_stale": True,
                },
            )

        await self.sse.emit(
            project_id,
            "project_status",
            {
                "project_id": project.id,
                "status": project.status,
                "report_stale": project.report_stale,
                "sync_pending": project.sync_pending,
                "prompt_safety_status": project.prompt_safety_status,
                "prompt_safety_violations_count": project.prompt_safety_violations_count,
            },
        )

        await self.sse.emit(project_id, "project_stats", self._build_project_stats(project))
        if heuristic_changed:
            await self.sse.emit(
                project_id,
                "heuristic_links_updated",
                {
                    "project_id": project.id,
                    "heuristic_links_added": heuristic_added,
                },
            )
        await self.sse.emit(
            project_id,
            "visualization_model_ready",
            {"project_id": project.id, "reason": "interview_processed"},
        )

        return {
            "status": "processed",
            "conversation_id": conversation_id,
            "interview_id": interview.id,
            "script_version": new_script.version,
            "sync_pending": project.sync_pending,
            "project_status": project.status,
            "report_stale": project.report_stale,
        }

    def _apply_analysis_result(self, project, result: AnalysisResult) -> None:
        proposition_index = {p.id: p for p in project.proposition_store}

        for evidence in result.new_evidence:
            if not evidence.id or any(e.id == evidence.id for e in project.evidence_store):
                evidence.id = self.project_service.next_evidence_id(project)
            if not str(evidence.quote_english or "").strip():
                if str(evidence.language or "").lower().startswith("en"):
                    evidence.quote_english = evidence.quote
                    evidence.translation_status = "native_en"
                else:
                    evidence.translation_status = "pending"
            project.evidence_store.append(evidence)

        for new_prop in result.new_propositions:
            if not new_prop.id or new_prop.id in proposition_index:
                new_prop.id = self.project_service.next_proposition_id(project)
            proposition_index[new_prop.id] = new_prop
            project.proposition_store.append(new_prop)

        evidence_index = {e.id: e for e in project.evidence_store}

        for mapping in result.evidence_mappings + result.retroactive_mappings:
            prop = proposition_index.get(mapping.proposition_id)
            evidence = evidence_index.get(mapping.evidence_id)
            if not prop or not evidence:
                continue
            if mapping.relationship == "supports":
                if evidence.id not in prop.supporting_evidence:
                    prop.supporting_evidence.append(evidence.id)
                if evidence.id in prop.contradicting_evidence:
                    prop.contradicting_evidence.remove(evidence.id)
                if evidence.id in prop.heuristic_supporting_evidence:
                    prop.heuristic_supporting_evidence.remove(evidence.id)
            elif mapping.relationship == "contradicts":
                if evidence.id not in prop.contradicting_evidence:
                    prop.contradicting_evidence.append(evidence.id)
                if evidence.id in prop.supporting_evidence:
                    prop.supporting_evidence.remove(evidence.id)
                if evidence.id in prop.heuristic_supporting_evidence:
                    prop.heuristic_supporting_evidence.remove(evidence.id)

        for update in result.proposition_updates:
            prop = proposition_index.get(update.id)
            if prop is None:
                continue
            prop.confidence = max(0.0, min(1.0, update.new_confidence))
            prop.status = update.new_status

        for prune_id in result.prunes:
            prop = proposition_index.get(prune_id)
            if prop:
                prop.status = "weak"

        project.metrics.convergence_score = result.metrics.convergence_score
        project.metrics.novelty_rate = result.metrics.novelty_rate
        project.metrics.mode = result.metrics.mode

    async def emit_analysis_events(self, project_id: str, result: AnalysisResult) -> None:
        for evidence in result.new_evidence:
            await self.sse.emit(project_id, "new_evidence", evidence.model_dump(mode="json"))

        for update in result.proposition_updates:
            await self.sse.emit(project_id, "proposition_updated", update.model_dump(mode="json"))

        for new_prop in result.new_propositions:
            await self.sse.emit(project_id, "new_proposition", new_prop.model_dump(mode="json"))

    def _build_project_stats(self, project) -> dict[str, Any]:
        return {
            "project_id": project.id,
            "status": project.status,
            "participants": len(project.interview_store),
            "interviews_count": len(project.interview_store),
            "evidence_count": len(project.evidence_store),
            "propositions_count": len(project.proposition_store),
            "active_propositions_count": len(
                [p for p in project.proposition_store if p.status not in {"weak", "merged"}]
            ),
            "convergence_score": project.metrics.convergence_score,
            "novelty_rate": project.metrics.novelty_rate,
            "mode": project.metrics.mode,
            "report_stale": project.report_stale,
            "report_generation_mode": project.report_generation_mode,
            "report_fallback_reason": project.report_fallback_reason,
            "prompt_safety_status": project.prompt_safety_status,
            "prompt_safety_violations_count": project.prompt_safety_violations_count,
        }
