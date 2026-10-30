"""Thin wrapper around the Neo4j driver for writing and querying oncology triples."""

import os
from typing import List, Optional, get_args

from langchain_core.documents import Document
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.graph_extraction import RelationType

ALL_RELATION_TYPES = get_args(RelationType)

# Read lazily inside get_driver(), not as module-level constants — a
# module-level os.environ.get() is evaluated at import time, which silently
# bakes in an empty password whenever this module gets imported before the
# caller's load_dotenv() runs (an easy-to-hit import-order footgun).

# One Cypher template per (subject_type, object_type) pair keeps labels
# static in the query (Neo4j doesn't allow parameterized labels), while still
# using MERGE so re-running extraction is idempotent rather than duplicating
# nodes/relationships.
#
# Nodes are keyed on a lowercased `key` rather than the raw `name` — 385
# independent LLM extraction calls won't capitalize the same entity
# identically every time ("Breast Cancer" vs "breast cancer"), and matching
# on `key` collapses those into one node instead of fragmenting the graph.
# `name` (whichever casing was first seen) is kept only for display.
_MERGE_TEMPLATE = (
    "MERGE (s:{subject_type} {{key: $subject_key}}) ON CREATE SET s.name = $subject "
    "MERGE (o:{object_type} {{key: $object_key}}) ON CREATE SET o.name = $object "
    "MERGE (s)-[r:{relation} {{source: $source}}]->(o)"
)


def get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


def write_triples(driver, triples: List[dict]) -> None:
    with driver.session() as session:
        for t in triples:
            query = _MERGE_TEMPLATE.format(
                subject_type=t["subject_type"],
                object_type=t["object_type"],
                relation=t["relation"],
            )
            session.run(
                query,
                subject=t["subject"],
                subject_key=t["subject"].lower(),
                object=t["object"],
                object_key=t["object"].lower(),
                source=t["source"],
            )


def all_entity_names(driver) -> List[str]:
    """Every node name in the graph, longest first (so multi-word entities like
    'Non-Small Cell Lung Cancer' match before a shorter substring like 'Lung')."""
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN DISTINCT n.name AS name")
        names = [record["name"] for record in result if record["name"]]
    return sorted(set(names), key=len, reverse=True)


def match_entities_in_text(all_names: List[str], text: str) -> List[str]:
    """Case-insensitive substring match of known graph entity names against text.

    `all_names` must already be sorted longest-first (see all_entity_names) so
    e.g. 'Non-Small Cell Lung Cancer' is matched, and consumed, before the
    shorter 'Lung Cancer' or 'Lung' would also match the same span.
    """
    lowered = text.lower()
    matched = []
    for name in all_names:
        if name.lower() in lowered:
            matched.append(name)
    return matched


def facts_for_entities(driver, entity_names: List[str], per_relation_limit: int = 5) -> List[dict]:
    """One-hop relationships (either direction) touching any of the given entities.

    Capped PER relation type, not globally — a matched Cancer entity easily has
    50+ USED_TO_TREAT edges (many drugs) but only a handful of INCREASES_RISK_OF
    edges (genes/risk factors); a single global LIMIT would let the prolific
    relation type crowd out the others entirely rather than sample across all
    the kinds of facts actually connected to the entity.
    """
    if not entity_names:
        return []
    facts = []
    with driver.session() as session:
        for relation in ALL_RELATION_TYPES:
            result = session.run(
                f"MATCH (a)-[r:{relation}]-(b) WHERE a.name IN $names "
                "RETURN DISTINCT a.name AS a, b.name AS b, "
                "startNode(r).name AS from_node, r.source AS source "
                "LIMIT $limit",
                names=entity_names,
                limit=per_relation_limit,
            )
            facts.extend(
                {
                    "subject": record["from_node"],
                    "relation": relation,
                    "object": record["b"] if record["from_node"] == record["a"] else record["a"],
                    "source": record["source"],
                }
                for record in result
            )
    return facts


class GraphFactsProvider:
    """App-facing entry point: question text in, graph-sourced Documents out.

    Fails soft — if Neo4j isn't reachable (not started yet, stopped, etc.) this
    returns an empty list rather than breaking the chat endpoint, since the
    vector-search path is still fully functional on its own.
    """

    def __init__(self):
        self._driver = None
        self._entity_names: Optional[List[str]] = None

    def _ensure_connected(self) -> bool:
        if self._driver is not None:
            return True
        try:
            driver = get_driver()
            driver.verify_connectivity()
        except (ServiceUnavailable, Neo4jError, OSError):
            return False
        self._driver = driver
        return True

    def get_facts(self, question: str, per_relation_limit: int = 5) -> List[Document]:
        if not self._ensure_connected():
            return []
        try:
            if self._entity_names is None:
                self._entity_names = all_entity_names(self._driver)
            matched = match_entities_in_text(self._entity_names, question)
            facts = facts_for_entities(self._driver, matched, per_relation_limit=per_relation_limit)
        except (ServiceUnavailable, Neo4jError):
            return []

        return [
            Document(
                page_content=f"{f['subject']} {f['relation'].replace('_', ' ').lower()} {f['object']}.",
                metadata={
                    "source": f"Neo4j knowledge graph ({f['source']})",
                    "content_type": "graph_fact",
                    # Structured fields (not just the rendered sentence) so a
                    # reviewer can delete this *exact* relationship later
                    # without having to re-parse prose back into a triple.
                    "subject": f["subject"],
                    "relation": f["relation"],
                    "object": f["object"],
                    "graph_source": f["source"],
                },
            )
            for f in facts
        ]


def delete_relationship(driver, subject: str, relation: str, obj: str, source: str) -> bool:
    """Delete one specific relationship, matched exactly as extraction wrote it.

    Used by the clinician review flow: a flagged graph fact gets removed at
    the source rather than just hidden, so it can never surface again.
    """
    if relation not in ALL_RELATION_TYPES:
        return False
    with driver.session() as session:
        result = session.run(
            f"MATCH (a {{key: $subject_key}})-[r:{relation} {{source: $source}}]->(b {{key: $object_key}}) "
            "DELETE r RETURN count(r) AS deleted",
            subject_key=subject.lower(),
            object_key=obj.lower(),
            source=source,
        )
        record = result.single()
        return bool(record and record["deleted"] > 0)
