# Medical Chatbot

A multi-department medical Q&A platform: a general assistant plus a specialized
**Cancer Department** agent for oncology clinicians. Built with LangChain, Pinecone
(vector search), Neo4j (knowledge graph), a live ClinicalTrials.gov tool, and a
Flask UI with a built-in clinician feedback loop.

## Architecture

```
                         ┌──────────────────┐
                         │   Flask app.py    │
                         └─────────┬────────┘
              ┌──────────────────────┼──────────────────────┐
      ┌───────▼────────┐                              ┌──────▼────────────┐
      │ General (/get)  │                              │ Cancer (/cancer/get)│
      │ plain RAG chain │                              │ RAG + graph + tool  │
      └───────┬────────┘                              └──────┬─────────────┘
              │                                              │
        ┌─────▼─────┐                          ┌─────────────┼─────────────┐
        │ Pinecone   │                          │             │             │
        │ (default   │                    ┌─────▼─────┐ ┌─────▼─────┐ ┌────▼──────────┐
        │ namespace) │                    │ Pinecone   │ │  Neo4j     │ │ ClinicalTrials│
        └───────────┘                    │ ('cancer'  │ │  (graph)   │ │ .gov (live)   │
                                          │ namespace) │ │            │ │               │
                                          └───────────┘ └───────────┘ └───────────────┘
```

Every Cancer Department answer merges three retrieval modes into one context
before generating: **vector search** (semantic, prose-level — "how is X
managed"), **the knowledge graph** (exact, relationship-level — "what treats
X"), and a **live tool call** to ClinicalTrials.gov when the question is about
trial availability, which static text can never answer correctly.

## Departments

### General (`/`, `/get`)
Plain RAG over whatever PDFs are in `data/`. Patient-facing tone.

### Cancer Department (`/cancer`, `/cancer/get`)
Clinician-facing. Knowledge base is 6 NCI PDQ® (Physician Data Query)
clinical summaries — the official, public-domain U.S. government cancer
guideline source, written for health professionals, not patients:

- Breast Cancer: Treatment, Screening, Prevention, Genetics (Breast & Gynecologic)
- Non-Small Cell Lung Cancer: Treatment
- Colon Cancer: Treatment

Embedded with **MedCPT** (`ncbi/MedCPT-Query-Encoder` / `-Article-Encoder`),
a biomedical-domain dual-encoder retrieval model — not a general-purpose
sentence embedder.

## Safety layer

Every incoming message goes through escalation → confidence gate → citations,
in that order, before an answer is generated (`src/safety.py`):

1. **Escalation** — deterministic keyword check, bypasses the LLM entirely.
   The general department checks for broad medical emergencies; the Cancer
   department's list is narrower (self-harm only), since a clinician
   discussing a patient's chest pain or dyspnea clinically shouldn't be
   blocked the way a patient reporting their own symptoms should be.
2. **Confidence gate** — refuses to guess if nothing relevant was retrieved
   *and* (for Cancer) the trial-search tool wasn't invoked either. Giving the
   agent a tool doesn't get to silently bypass this refusal.
3. **Citations** — every answer returns the exact chunks / graph facts / tool
   output it was grounded in, so a clinician can verify it.

This is a heuristic baseline, not a validated clinical safety system —
before any real deployment, have the red-flag list and confidence threshold
reviewed by a licensed clinician.

## The knowledge graph

`build_graph.py` extracts `(subject, relation, object)` triples from the PDQ
text via a schema-constrained LLM call (`src/graph_extraction.py`) — not an
imported medical ontology (UMLS/SNOMED-CT require a license account; NCI
Thesaurus is public but far larger than this scope needs). Schema:

```
(:Gene)       -[:INCREASES_RISK_OF]-> (:Cancer)
(:RiskFactor) -[:INCREASES_RISK_OF]-> (:Cancer)
(:Drug)       -[:USED_TO_TREAT]->     (:Cancer | :Stage)
(:Cancer)     -[:HAS_STAGE]->         (:Stage)     — TNM/numeric staging only
(:Cancer)     -[:HAS_SUBTYPE]->       (:Subtype)   — molecular/receptor classification
```

Nodes are keyed on a lowercased `key` (display `name` keeps original casing)
so independent extraction calls that capitalize an entity differently still
merge into one node instead of fragmenting the graph.

**Known limitation:** extraction isn't perfectly reliable — expect an
occasional reversed triple (e.g. "Stage I has stage Breast Cancer") or a
misclassified entity. This is exactly what the feedback/review loop below is
for: fixing it with human judgment instead of paying for more re-extraction.

## Clinician feedback loop

Every Cancer Department answer has 👍/👎 on the answer and a 🚩 Flag button on
each individual source (vector chunk, graph fact, or trial result). Feedback
is stored locally in SQLite (`feedback.db`, gitignored — it's runtime state,
not source).

Review it at **`/cancer/review`**. For a flagged graph fact, "Delete from
graph" removes that *exact* relationship from Neo4j permanently — it can
never resurface, because it's gone from the source, not just hidden in the
UI. This is the intended way to fix the known extraction-noise limitation
above over time, without spending more on LLM re-extraction.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and start Neo4j (for the Cancer department's graph)

```bash
brew install neo4j
neo4j-admin dbms set-initial-password "<a strong password>"
brew services start neo4j
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

- `PINECONE_API_KEY` — from your [Pinecone](https://www.pinecone.io/) account (free tier works).
- `PINECONE_INDEX_NAME` — created automatically if missing.
- `OPENAI_API_KEY` — used for the chat LLM (`gpt-4o-mini`) and graph extraction.
  **Extraction and chat both cost money** — `build_graph.py` makes hundreds of
  LLM calls; budget accordingly before re-running it on a larger corpus.
- `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` — from step 2.
- `RETRIEVAL_K`, `CONFIDENCE_THRESHOLD` — safety-layer tuning.

### 4. Add source documents

- General department: PDFs in `data/`.
- Cancer department: PDFs or `.txt` files in `data/cancer/`.

### 5. Build the indexes

```bash
python store_index.py                                    # general department
DATA_DIR=data/cancer/ PINECONE_NAMESPACE=cancer python store_index.py   # Cancer dept vectors
DATA_DIR=data/cancer/ python build_graph.py               # Cancer dept graph (costs OpenAI $)
```

Both are safe to re-run (Pinecone upsert is additive per-run though — clear
the namespace first if re-indexing the *same* files to avoid duplicates;
Neo4j's MERGE is naturally idempotent).

### 6. Run

```bash
python app.py
```

- General assistant: http://localhost:8080/
- Cancer Department: http://localhost:8080/cancer
- Feedback review dashboard: http://localhost:8080/cancer/review

## Project structure

```
├── app.py                    # Flask app: both departments, tool-calling, feedback routes
├── build_graph.py            # Extracts triples from data/cancer/ into Neo4j
├── store_index.py            # Builds/refreshes the Pinecone vector index
├── src/
│   ├── helper.py              # PDF/text loading, chunking, embeddings
│   ├── medcpt_embeddings.py   # MedCPT dual-encoder LangChain Embeddings wrapper
│   ├── prompt.py               # System prompts (general + cancer/clinician)
│   ├── safety.py               # Red-flag escalation + canned safety responses
│   ├── multimodal.py           # Table/image extraction from PDFs (general dept)
│   ├── graph_extraction.py    # LLM triple extraction (schema + prompt)
│   ├── graph_store.py         # Neo4j read/write + GraphFactsProvider
│   ├── clinical_trials.py     # ClinicalTrials.gov API tool
│   └── feedback_store.py      # SQLite feedback storage
├── templates/
│   ├── chat.html               # Chat UI (shared by both departments)
│   └── review.html             # Clinician feedback review dashboard
├── static/
│   ├── style.css                # Chat UI styling
│   └── review.css               # Review dashboard styling
├── data/                      # General department source PDFs
├── data/cancer/               # Cancer department source PDFs/text (NCI PDQ)
└── feedback.db                # Clinician feedback (SQLite, gitignored)
```

## Disclaimer

This chatbot is for informational purposes only and is not a substitute for
professional medical advice, diagnosis, or treatment. The Cancer Department
supports, but does not replace, clinical judgment, tumor board review, and
current full-text guidelines.
