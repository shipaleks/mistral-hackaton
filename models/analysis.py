from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from models.evidence import Evidence
from models.proposition import Proposition, PropositionStatus


class EvidenceMapping(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evidence_id: str
    proposition_id: str
    relationship: Literal["supports", "contradicts"]


class PropositionUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    new_confidence: float
    new_status: PropositionStatus


class MergeProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_ids: list[str] = Field(default_factory=list)
    merged_proposition: Proposition | None = None


class AnalysisMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    convergence_score: float = 0.0
    novelty_rate: float = 1.0
    mode: Literal["divergent", "convergent"] = "divergent"


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    new_evidence: list[Evidence] = Field(default_factory=list)
    evidence_mappings: list[EvidenceMapping] = Field(default_factory=list)
    new_propositions: list[Proposition] = Field(default_factory=list)
    retroactive_mappings: list[EvidenceMapping] = Field(default_factory=list)
    proposition_updates: list[PropositionUpdate] = Field(default_factory=list)
    merges: list[MergeProposal] = Field(default_factory=list)
    prunes: list[str] = Field(default_factory=list)
    metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics)
