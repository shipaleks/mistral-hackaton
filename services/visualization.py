from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from models.project import ProjectState


def _tokenize(text: str) -> set[str]:
    try:
        tokens = re.findall(r"[\w]+", str(text or "").lower(), flags=re.UNICODE)
    except re.error:
        tokens = re.findall(r"[a-z0-9а-яё]+", str(text or "").lower())
    return {t for t in tokens if len(t) > 1}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union


def _similarity_score(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    proposition_overlap = _jaccard(a["propositions"], b["propositions"])
    tag_overlap = _jaccard(a["tags"], b["tags"])
    fmo_overlap = _jaccard(a["fmo_tokens"], b["fmo_tokens"])
    score = 0.45 * proposition_overlap + 0.35 * tag_overlap + 0.20 * fmo_overlap
    return {
        "score": score,
        "proposition_overlap": proposition_overlap,
        "tag_overlap": tag_overlap,
        "fmo_overlap": fmo_overlap,
    }


def build_hypothesis_map(project: ProjectState) -> dict[str, Any]:
    evidence_by_id = {e.id: e for e in project.evidence_store}
    proposition_by_id = {p.id: p for p in project.proposition_store}
    evidence_to_props: dict[str, set[str]] = {e.id: set() for e in project.evidence_store}

    for proposition in project.proposition_store:
        for evidence_id in proposition.supporting_evidence + proposition.contradicting_evidence:
            if evidence_id in evidence_to_props:
                evidence_to_props[evidence_id].add(proposition.id)

    evidence_features: dict[str, dict[str, Any]] = {}
    for evidence in project.evidence_store:
        feature = {
            "id": evidence.id,
            "tags": {str(tag).lower() for tag in evidence.tags},
            "fmo_tokens": _tokenize(f"{evidence.factor} {evidence.mechanism} {evidence.outcome}"),
            "propositions": set(evidence_to_props.get(evidence.id, set())),
            "interview_id": evidence.interview_id,
            "factor": evidence.factor,
            "mechanism": evidence.mechanism,
            "outcome": evidence.outcome,
            "quote": evidence.quote,
            "language": evidence.language,
        }
        evidence_features[evidence.id] = feature

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for proposition in project.proposition_store:
        nodes.append(
            {
                "id": proposition.id,
                "type": "hypothesis",
                "label": proposition.factor,
                "status": proposition.status,
                "confidence": proposition.confidence,
                "support_count": len(proposition.supporting_evidence),
                "contradict_count": len(proposition.contradicting_evidence),
            }
        )

    for evidence in project.evidence_store:
        nodes.append(
            {
                "id": evidence.id,
                "type": "evidence",
                "label": evidence.quote[:96],
                "interview_id": evidence.interview_id,
                "language": evidence.language,
                "factor": evidence.factor,
                "mechanism": evidence.mechanism,
                "outcome": evidence.outcome,
                "mapped_hypotheses": sorted(list(evidence_to_props.get(evidence.id, set()))),
            }
        )

    for proposition in project.proposition_store:
        for evidence_id in proposition.supporting_evidence:
            if evidence_id not in evidence_by_id:
                continue
            edges.append(
                {
                    "source": proposition.id,
                    "target": evidence_id,
                    "relation": "supports",
                    "style": "solid",
                    "score": 1.0,
                }
            )
        for evidence_id in proposition.contradicting_evidence:
            if evidence_id not in evidence_by_id:
                continue
            edges.append(
                {
                    "source": proposition.id,
                    "target": evidence_id,
                    "relation": "contradicts",
                    "style": "solid",
                    "score": 1.0,
                }
            )

    unassigned_ids = [eid for eid, props in evidence_to_props.items() if not props]

    clusters: list[dict[str, Any]] = []
    visited: set[str] = set()
    cluster_counter = 0

    for evidence_id in unassigned_ids:
        if evidence_id in visited:
            continue
        cluster_counter += 1
        queue = [evidence_id]
        visited.add(evidence_id)
        members = [evidence_id]

        while queue:
            current = queue.pop(0)
            current_feature = evidence_features.get(current)
            if not current_feature:
                continue
            for candidate_id in unassigned_ids:
                if candidate_id in visited:
                    continue
                candidate_feature = evidence_features.get(candidate_id)
                if not candidate_feature:
                    continue
                if _similarity_score(current_feature, candidate_feature)["score"] >= 0.70:
                    visited.add(candidate_id)
                    queue.append(candidate_id)
                    members.append(candidate_id)

        token_counter: Counter[str] = Counter()
        for member_id in members:
            feature = evidence_features.get(member_id)
            if not feature:
                continue
            token_counter.update(feature["fmo_tokens"])
            token_counter.update(feature["tags"])

        top_tokens = [token for token, _ in token_counter.most_common(5)]
        label = " / ".join(top_tokens[:3]) if top_tokens else "emerging pattern"
        cluster_id = f"CLUSTER_{cluster_counter:03d}"

        nodes.append(
            {
                "id": cluster_id,
                "type": "candidate_cluster",
                "label": f"Candidate {cluster_counter}: {label}",
                "size": len(members),
                "evidence_ids": members,
            }
        )

        for member_id in members:
            edges.append(
                {
                    "source": cluster_id,
                    "target": member_id,
                    "relation": "candidate_member",
                    "style": "dotted",
                    "score": 1.0,
                }
            )

        potential_supporters: list[dict[str, Any]] = []
        cluster_tokens = set(top_tokens)
        for mapped_id, mapped_feature in evidence_features.items():
            if mapped_id in members:
                continue
            if not mapped_feature["propositions"]:
                continue
            compare_tokens = mapped_feature["fmo_tokens"].union(mapped_feature["tags"])
            score = _jaccard(cluster_tokens, compare_tokens)
            if score >= 0.45:
                potential_supporters.append(
                    {
                        "evidence_id": mapped_id,
                        "score": round(score, 3),
                    }
                )
                edges.append(
                    {
                        "source": cluster_id,
                        "target": mapped_id,
                        "relation": "potential_supporter",
                        "style": "dashed",
                        "score": round(score, 3),
                    }
                )

        potential_supporters = sorted(
            potential_supporters,
            key=lambda item: item["score"],
            reverse=True,
        )[:6]
        clusters.append(
            {
                "id": cluster_id,
                "label": label,
                "evidence_ids": members,
                "potential_supporters": potential_supporters,
            }
        )

    unassigned_pool = [
        {
            "evidence_id": evidence_id,
            "factor": evidence_features[evidence_id]["factor"],
            "mechanism": evidence_features[evidence_id]["mechanism"],
            "outcome": evidence_features[evidence_id]["outcome"],
            "quote": evidence_features[evidence_id]["quote"][:160],
        }
        for evidence_id in unassigned_ids
        if evidence_id in evidence_features
    ]

    supports_count = sum(1 for edge in edges if edge["relation"] == "supports")
    contradicts_count = sum(1 for edge in edges if edge["relation"] == "contradicts")

    return {
        "project_id": project.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "hypotheses": len(project.proposition_store),
            "evidence": len(project.evidence_store),
            "supports_edges": supports_count,
            "contradicts_edges": contradicts_count,
            "unassigned_evidence": len(unassigned_ids),
            "candidate_clusters": len(clusters),
        },
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "unassigned_pool": unassigned_pool,
        "similarity_formula": {
            "threshold": 0.70,
            "weights": {
                "proposition_overlap": 0.45,
                "tag_jaccard": 0.35,
                "fmo_token_overlap": 0.20,
            },
        },
    }
