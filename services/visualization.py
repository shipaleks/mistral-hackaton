from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from models.project import ProjectState


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "about",
    "your",
    "their",
    "have",
    "were",
    "was",
    "are",
    "not",
    "but",
    "you",
    "they",
    "them",
    "our",
    "ours",
    "its",
    "had",
    "has",
    "been",
    "just",
    "very",
    "more",
    "less",
    "some",
    "such",
    "than",
    "then",
    "also",
    "over",
    "under",
    "made",
    "make",
    "using",
    "used",
    "when",
    "what",
    "where",
    "while",
    "into",
    "a",
    "an",
    "of",
    "to",
    # Russian stopwords
    "это", "как", "что", "для", "при", "они", "она", "его",
    "мне", "мой", "мои", "все", "или", "уже", "так", "тоже",
    "вот", "где", "там", "тут", "ещё", "был", "была", "были",
    "быть", "очень", "когда", "если", "чтобы", "этот", "эта",
    "эти", "тот", "того", "между", "через", "после", "перед",
    "более", "менее", "также", "только", "можно", "нужно",
    "надо", "потому", "самый", "самая", "самое",
}


def _tokenize(text: str) -> set[str]:
    try:
        tokens = re.findall(r"[\w]+", str(text or "").lower(), flags=re.UNICODE)
    except re.error:
        tokens = re.findall(r"[a-z0-9а-яё]+", str(text or "").lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    }


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union


def _evidence_feature(project: ProjectState, evidence_id: str, evidence_to_props: dict[str, set[str]]) -> dict[str, Any]:
    evidence = next((item for item in project.evidence_store if item.id == evidence_id), None)
    if evidence is None:
        return {
            "id": evidence_id,
            "tags": set(),
            "fmo_tokens": set(),
            "quote_tokens": set(),
            "propositions": set(),
            "interview_id": "",
        }

    quote_for_view = str(evidence.quote_english or "").strip() or str(evidence.quote or "")
    return {
        "id": evidence.id,
        "tags": {str(tag).lower() for tag in evidence.tags},
        "fmo_tokens": _tokenize(f"{evidence.factor} {evidence.mechanism} {evidence.outcome}"),
        "quote_tokens": _tokenize(quote_for_view),
        "propositions": set(evidence_to_props.get(evidence.id, set())),
        "interview_id": evidence.interview_id,
        "factor": evidence.factor,
        "mechanism": evidence.mechanism,
        "outcome": evidence.outcome,
        "quote": evidence.quote,
        "quote_english": evidence.quote_english,
        "translation_status": evidence.translation_status,
        "language": evidence.language,
    }


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


def _evidence_to_hypothesis_score(evidence_feature: dict[str, Any], proposition: Any) -> dict[str, float]:
    proposition_tokens = _tokenize(
        f"{proposition.factor} {proposition.mechanism} {proposition.outcome}"
    )
    proposition_overlap = _jaccard(evidence_feature["fmo_tokens"], proposition_tokens)
    tag_overlap = _jaccard(evidence_feature["tags"].union(evidence_feature["fmo_tokens"]), proposition_tokens)
    fmo_overlap = _jaccard(evidence_feature["quote_tokens"].union(evidence_feature["fmo_tokens"]), proposition_tokens)
    score = 0.45 * proposition_overlap + 0.35 * tag_overlap + 0.20 * fmo_overlap
    return {
        "score": score,
        "proposition_overlap": proposition_overlap,
        "tag_overlap": tag_overlap,
        "fmo_overlap": fmo_overlap,
    }


def compute_heuristic_links(
    project: ProjectState,
    threshold: float = 0.70,
) -> dict[str, list[dict[str, Any]]]:
    evidence_to_props: dict[str, set[str]] = {e.id: set() for e in project.evidence_store}
    for proposition in project.proposition_store:
        for evidence_id in proposition.supporting_evidence + proposition.contradicting_evidence:
            if evidence_id in evidence_to_props:
                evidence_to_props[evidence_id].add(proposition.id)

    unassigned_ids = [eid for eid, props in evidence_to_props.items() if not props]
    evidence_features = {
        evidence_id: _evidence_feature(project, evidence_id, evidence_to_props)
        for evidence_id in evidence_to_props
    }

    links_by_prop: dict[str, list[dict[str, Any]]] = {}
    for proposition in project.proposition_store:
        if proposition.status in {"weak", "merged"}:
            continue

        confirmed_count = len(proposition.supporting_evidence) + len(proposition.contradicting_evidence)
        if confirmed_count > 4:
            continue

        candidates: list[dict[str, Any]] = []
        for evidence_id in unassigned_ids:
            if evidence_id in proposition.supporting_evidence or evidence_id in proposition.contradicting_evidence:
                continue
            feature = evidence_features.get(evidence_id)
            if feature is None:
                continue
            score = _evidence_to_hypothesis_score(feature, proposition)
            if score["score"] < threshold:
                continue
            candidates.append(
                {
                    "evidence_id": evidence_id,
                    "score": round(score["score"], 3),
                    "proposition_overlap": round(score["proposition_overlap"], 3),
                    "tag_overlap": round(score["tag_overlap"], 3),
                    "fmo_overlap": round(score["fmo_overlap"], 3),
                }
            )

        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)
            links_by_prop[proposition.id] = candidates[:3]

    return links_by_prop


def apply_heuristic_links(
    project: ProjectState,
    threshold: float = 0.70,
) -> tuple[bool, int]:
    links_by_prop = compute_heuristic_links(project=project, threshold=threshold)
    changed = False
    added_count = 0

    for proposition in project.proposition_store:
        suggestions = links_by_prop.get(proposition.id, [])
        new_ids = [item["evidence_id"] for item in suggestions]
        if new_ids == proposition.heuristic_supporting_evidence:
            continue

        previous = set(proposition.heuristic_supporting_evidence)
        current = set(new_ids)
        added_count += max(0, len(current.difference(previous)))
        proposition.heuristic_supporting_evidence = new_ids
        changed = True

    return changed, added_count


def _clean_candidate_label(tokens: list[str]) -> str:
    meaningful = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
    top = meaningful[:3]
    if not top:
        return "emerging pattern"
    return " / ".join(top)


def build_hypothesis_map(project: ProjectState) -> dict[str, Any]:
    evidence_by_id = {e.id: e for e in project.evidence_store}
    evidence_to_props: dict[str, set[str]] = {e.id: set() for e in project.evidence_store}

    for proposition in project.proposition_store:
        for evidence_id in proposition.supporting_evidence + proposition.contradicting_evidence:
            if evidence_id in evidence_to_props:
                evidence_to_props[evidence_id].add(proposition.id)

    evidence_features: dict[str, dict[str, Any]] = {
        evidence.id: _evidence_feature(project, evidence.id, evidence_to_props)
        for evidence in project.evidence_store
    }

    heuristic_links = compute_heuristic_links(project)
    heuristic_count_by_prop = {
        proposition_id: len(items) for proposition_id, items in heuristic_links.items()
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    status_legend: dict[str, int] = {
        "untested": 0,
        "exploring": 0,
        "confirmed": 0,
        "challenged": 0,
        "saturated": 0,
        "weak": 0,
        "merged": 0,
    }

    for proposition in project.proposition_store:
        status_legend[proposition.status] = status_legend.get(proposition.status, 0) + 1
        nodes.append(
            {
                "id": proposition.id,
                "type": "hypothesis",
                "label": proposition.factor,
                "status": proposition.status,
                "confidence": proposition.confidence,
                "support_count": len(proposition.supporting_evidence),
                "contradict_count": len(proposition.contradicting_evidence),
                "heuristic_support_count": heuristic_count_by_prop.get(proposition.id, 0),
                "first_seen_interview": proposition.first_seen_interview,
                "last_updated_interview": proposition.last_updated_interview,
                "fmo": {
                    "factor": proposition.factor,
                    "mechanism": proposition.mechanism,
                    "outcome": proposition.outcome,
                },
            }
        )

    project_language = getattr(project, "language", "en") or "en"
    for evidence in project.evidence_store:
        if project_language == "ru":
            display_quote = str(evidence.quote or "").strip()
            if not display_quote:
                display_quote = "Перевод ожидается"
        else:
            display_quote = str(evidence.quote_english or "").strip()
            if not display_quote:
                display_quote = "Translation pending"

        nodes.append(
            {
                "id": evidence.id,
                "type": "evidence",
                "label": display_quote[:120],
                "display_quote_en": display_quote,
                "interview_id": evidence.interview_id,
                "language": evidence.language,
                "translation_status": evidence.translation_status,
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
                    "source_type": "llm",
                    "style": "solid",
                    "score": 1.0,
                    "explanation": "Mapped by analyst as supporting evidence.",
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
                    "source_type": "llm",
                    "style": "solid",
                    "score": 1.0,
                    "explanation": "Mapped by analyst as contradicting evidence.",
                }
            )

        heuristic_targets = heuristic_links.get(proposition.id, [])
        for heuristic_item in heuristic_targets:
            evidence_id = heuristic_item["evidence_id"]
            if evidence_id in proposition.supporting_evidence or evidence_id in proposition.contradicting_evidence:
                continue
            explanation = (
                "Heuristic match: "
                f"score={heuristic_item['score']} "
                f"(proposition={heuristic_item['proposition_overlap']}, "
                f"tags={heuristic_item['tag_overlap']}, fmo={heuristic_item['fmo_overlap']})."
            )
            edges.append(
                {
                    "source": proposition.id,
                    "target": evidence_id,
                    "relation": "supports",
                    "source_type": "heuristic",
                    "style": "dashed",
                    "score": heuristic_item["score"],
                    "explanation": explanation,
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

        top_tokens = [token for token, _ in token_counter.most_common(6)]
        label = _clean_candidate_label(top_tokens)
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
                    "source_type": "heuristic",
                    "style": "dotted",
                    "score": 1.0,
                    "explanation": "Evidence grouped into candidate cluster by similarity threshold 0.70.",
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
                        "source_type": "heuristic",
                        "style": "dashed",
                        "score": round(score, 3),
                        "explanation": "Potential historical supporter based on token overlap >= 0.45.",
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
            "quote": str(
                evidence_features[evidence_id]["quote_english"]
                or "Translation pending"
            )[:160],
        }
        for evidence_id in unassigned_ids
        if evidence_id in evidence_features
    ]

    validated_hypotheses = [
        proposition
        for proposition in project.proposition_store
        if len(proposition.supporting_evidence) + len(proposition.contradicting_evidence) > 0
    ]
    unvalidated_hypotheses = [
        {
            "id": proposition.id,
            "label": proposition.factor,
            "status": proposition.status,
            "confidence": proposition.confidence,
            "first_seen_interview": proposition.first_seen_interview,
            "last_updated_interview": proposition.last_updated_interview,
            "heuristic_support_count": heuristic_count_by_prop.get(proposition.id, 0),
        }
        for proposition in project.proposition_store
        if len(proposition.supporting_evidence) + len(proposition.contradicting_evidence) == 0
    ]

    supports_count = sum(1 for edge in edges if edge["relation"] == "supports" and edge["source_type"] == "llm")
    contradicts_count = sum(1 for edge in edges if edge["relation"] == "contradicts")
    heuristic_support_count = sum(
        1
        for edge in edges
        if edge["relation"] == "supports" and edge["source_type"] == "heuristic"
    )

    latest_interview_idx = len(project.interview_store)
    new_latest_count = sum(
        1
        for proposition in project.proposition_store
        if proposition.first_seen_interview == latest_interview_idx and latest_interview_idx > 0
    )

    return {
        "project_id": project.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "hypotheses": len(project.proposition_store),
            "evidence": len(project.evidence_store),
            "supports_edges": supports_count,
            "contradicts_edges": contradicts_count,
            "heuristic_supports_edges": heuristic_support_count,
            "unassigned_evidence": len(unassigned_ids),
            "candidate_clusters": len(clusters),
            "validated_hypotheses": len(validated_hypotheses),
            "unvalidated_hypotheses": len(unvalidated_hypotheses),
        },
        "status_legend": status_legend,
        "progress_snapshot": {
            "latest_interview_index": latest_interview_idx,
            "new_hypotheses_latest_interview": new_latest_count,
            "total_hypotheses": len(project.proposition_store),
            "validated_hypotheses": len(validated_hypotheses),
            "unvalidated_hypotheses": len(unvalidated_hypotheses),
        },
        "unvalidated_hypotheses": unvalidated_hypotheses,
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
