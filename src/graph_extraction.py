"""LLM-based (subject, relation, object) triple extraction for the oncology graph.

Deliberately narrow: a fixed entity/relation vocabulary, not open-ended NER.
Constraining the schema is what keeps this from hallucinating relation types
that don't exist in the graph and can never be queried.
"""

from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

EntityType = Literal["Cancer", "Gene", "RiskFactor", "Drug", "Stage", "Subtype"]
RelationType = Literal["INCREASES_RISK_OF", "USED_TO_TREAT", "HAS_STAGE", "HAS_SUBTYPE"]


class Triple(BaseModel):
    subject: str = Field(description="Entity name, as written in the text (e.g. 'BRCA1', 'Tamoxifen').")
    subject_type: EntityType
    relation: RelationType
    object: str = Field(description="Entity name, as written in the text.")
    object_type: EntityType


class TripleList(BaseModel):
    triples: List[Triple] = Field(default_factory=list)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured facts from oncology clinical text into a fixed "
            "schema. Only extract a triple if the relationship is EXPLICITLY stated "
            "in the text — never infer, guess, or use outside knowledge. If nothing "
            "in the text fits the schema, return an empty list; that is a correct, "
            "expected answer for most chunks, not a failure.\n\n"
            "Allowed relations:\n"
            "- INCREASES_RISK_OF: a Gene or RiskFactor raises risk of a Cancer\n"
            "- USED_TO_TREAT: a Drug is used to treat a Cancer, a Stage, or a Subtype\n"
            "- HAS_STAGE: a Cancer has a named anatomical/TNM STAGE ONLY — e.g. "
            "'Stage I', 'Stage IIIA', 'T2', 'N0', 'M1', 'Stage 0'. Never put a "
            "molecular/receptor classification here.\n"
            "- HAS_SUBTYPE: a Cancer has a named molecular/receptor/histologic "
            "SUBTYPE — e.g. 'HER2-positive', 'Hormone receptor-positive', "
            "'Triple-negative', 'Ductal carcinoma in situ'. These describe what "
            "kind of tumor it is, not how far it has spread — never put these "
            "under HAS_STAGE even though the text may call them a 'stage' loosely.\n\n"
            "Use short, canonical entity names (e.g. 'BRCA1' not 'the BRCA1 gene "
            "pathogenic variant'; 'Breast Cancer' not 'breast cancer patients'). "
            "Use consistent capitalization for the same entity across calls: "
            "Title Case for cancer/disease names ('Breast Cancer', 'Non-Small "
            "Cell Lung Cancer'), and standard scientific casing for genes/drugs "
            "('BRCA1', 'Tamoxifen').",
        ),
        ("human", "{text}"),
    ]
)


def build_extractor(llm):
    """Return a Runnable that maps raw text -> TripleList via structured output."""
    structured_llm = llm.with_structured_output(TripleList)
    return EXTRACTION_PROMPT | structured_llm


def extract_triples(extractor, text: str, source: str) -> List[dict]:
    """Run the extractor on one chunk, tagging each triple with its source doc."""
    result: TripleList = extractor.invoke({"text": text})
    return [
        {
            "subject": t.subject.strip(),
            "subject_type": t.subject_type,
            "relation": t.relation,
            "object": t.object.strip(),
            "object_type": t.object_type,
            "source": source,
        }
        for t in result.triples
        if t.subject.strip() and t.object.strip()
    ]
