import os

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.clinical_trials import search_clinical_trials
from src.feedback_store import add_feedback, get_feedback, init_db, list_feedback, mark_resolved
from src.graph_store import GraphFactsProvider, delete_relationship, get_driver
from src.helper import download_hugging_face_embeddings
from src.prompt import cancer_system_prompt, system_prompt
from src.safety import (
    CANCER_EMERGENCY_RESPONSE,
    CANCER_LOW_CONFIDENCE_RESPONSE,
    EMERGENCY_RESPONSE,
    LOW_CONFIDENCE_RESPONSE,
    is_cancer_red_flag,
    is_red_flag,
)

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "3"))
# Pinecone cosine similarity is in [-1, 1]; below this, the retrieved context
# is too weak to ground an answer in, so we refuse rather than let the LLM
# generate something fluent but unsupported.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

app = Flask(__name__)
init_db()

embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


def make_department(namespace: str, system_prompt_text: str):
    """Wire up one department's retriever + QA chain against its Pinecone namespace."""
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace or None,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return docsearch, qa_chain


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_with_tools(user_input: str, docs: list, system_prompt_text: str, tools: list):
    """One round of real tool-calling: let the model decide whether to call a
    tool (e.g. live trial search) before answering, rather than always
    fetching it unconditionally like the graph facts. Capped at one round —
    good enough for a single tool, avoids open-ended agent-loop complexity.
    """
    tool_llm = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=system_prompt_text.format(context=format_docs(docs))),
        HumanMessage(content=user_input),
    ]
    response = tool_llm.invoke(messages)

    tool_sources = []
    if response.tool_calls:
        messages.append(response)
        tools_by_name = {t.name: t for t in tools}
        for call in response.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            tool_sources.append(
                {
                    "source": f"ClinicalTrials.gov ({call['args'].get('condition', '')})",
                    "page": None,
                    "excerpt": result,
                    "score": None,
                    "content_type": "clinical_trial",
                    "image_path": None,
                }
            )
        response = tool_llm.invoke(messages)

    return response.content, tool_sources


def handle_chat(
    user_input: str,
    docsearch,
    qa_chain,
    is_flag_fn,
    emergency_response: str,
    low_confidence_response: str,
    graph_provider: "GraphFactsProvider | None" = None,
    tools: "list | None" = None,
    system_prompt_text: "str | None" = None,
) -> dict:
    """Shared retrieve-then-answer flow: escalation, confidence gate, citations.

    When graph_provider is given, structured facts from Neo4j (exact,
    relationship-level — "what treats X") are merged in alongside the
    similarity-ranked chunks (semantic, prose-level — "how is X managed"),
    since the two retrieval modes cover different kinds of questions.

    When tools is given, the model itself decides whether to call one (e.g.
    live ClinicalTrials.gov search) before answering — unlike graph facts,
    which are always fetched, a tool call only makes sense for some questions.
    """

    # 1. Escalation: deterministic, runs before any retrieval or generation.
    if is_flag_fn(user_input):
        return {"answer": emergency_response, "sources": [], "confidence": "emergency"}

    # 2. Confidence gate: only generate if we actually found grounding for it.
    results = docsearch.similarity_search_with_score(user_input, k=RETRIEVAL_K)
    graph_docs = graph_provider.get_facts(user_input) if graph_provider else []
    has_static_grounding = bool(results and results[0][1] >= CONFIDENCE_THRESHOLD) or bool(graph_docs)

    if not has_static_grounding and not tools:
        return {"answer": low_confidence_response, "sources": [], "confidence": "low"}

    # 3. Citations: return the exact chunks (+ graph facts) the answer is grounded in.
    docs = [doc for doc, _score in results if _score >= CONFIDENCE_THRESHOLD] + graph_docs

    if tools:
        answer, tool_sources = answer_with_tools(user_input, docs, system_prompt_text, tools)
        # If there was no static grounding AND the model didn't call a tool
        # either, it has nothing real to answer from — refuse rather than let
        # a context-free LLM call improvise. Giving `tools` shouldn't quietly
        # bypass the refusal gate for questions the tool has nothing to do with.
        if not has_static_grounding and not tool_sources:
            return {"answer": low_confidence_response, "sources": [], "confidence": "low"}
    else:
        answer = qa_chain.invoke({"input": user_input, "context": docs})
        tool_sources = []

    sources = (
        [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:200].strip(),
                "score": round(float(score), 3),
                "content_type": doc.metadata.get("content_type", "text"),
                "image_path": doc.metadata.get("image_path"),
            }
            for doc, score in results
            if score >= CONFIDENCE_THRESHOLD
        ]
        + [
            {
                "source": doc.metadata.get("source", "Neo4j knowledge graph"),
                "page": None,
                "excerpt": doc.page_content,
                "score": None,
                "content_type": "graph_fact",
                "image_path": None,
                # Carried through so a clinician's "flag" can target this
                # exact relationship for deletion, not just the rendered text.
                "subject": doc.metadata.get("subject"),
                "relation": doc.metadata.get("relation"),
                "object": doc.metadata.get("object"),
                "graph_source": doc.metadata.get("graph_source"),
            }
            for doc in graph_docs
        ]
        + tool_sources
    )

    return {"answer": answer, "sources": sources, "confidence": "normal"}


general_docsearch, general_qa_chain = make_department("", system_prompt)
cancer_docsearch, cancer_qa_chain = make_department("cancer", cancer_system_prompt)
# Fails soft if Neo4j isn't running — see GraphFactsProvider._ensure_connected.
cancer_graph = GraphFactsProvider()


@app.route("/")
def index():
    return render_template("chat.html", title="Medical Chatbot")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"].strip()
    result = handle_chat(
        user_input,
        general_docsearch,
        general_qa_chain,
        is_red_flag,
        EMERGENCY_RESPONSE,
        LOW_CONFIDENCE_RESPONSE,
    )
    return jsonify(result)


@app.route("/cancer")
def cancer_index():
    return render_template(
        "chat.html",
        title="Cancer Department",
        endpoint="/cancer/get",
        feedback_endpoint="/cancer/feedback",
        placeholder="Ask an oncology question (NCI PDQ, clinician-facing)...",
    )


@app.route("/cancer/feedback", methods=["POST"])
def cancer_feedback():
    data = request.get_json(silent=True) or {}
    if not data.get("question") or not data.get("rating"):
        return jsonify({"ok": False, "error": "question and rating are required"}), 400

    item_meta = None
    if data.get("item_type") == "graph_fact":
        item_meta = {
            "subject": data.get("subject"),
            "relation": data.get("relation"),
            "object": data.get("object"),
            "graph_source": data.get("graph_source"),
        }

    add_feedback(
        question=data["question"],
        rating=data["rating"],
        answer=data.get("answer"),
        item_type=data.get("item_type"),
        item_source=data.get("item_source"),
        item_excerpt=data.get("item_excerpt"),
        item_meta=item_meta,
    )
    return jsonify({"ok": True})


@app.route("/cancer/review")
def cancer_review():
    return render_template("review.html", feedback=list_feedback(unresolved_only=True))


@app.route("/cancer/review/resolve", methods=["POST"])
def cancer_review_resolve():
    feedback_id = request.form.get("feedback_id", type=int)
    if feedback_id is not None:
        mark_resolved(feedback_id)
    return redirect(url_for("cancer_review"))


@app.route("/cancer/review/delete-graph-fact", methods=["POST"])
def cancer_review_delete_graph_fact():
    feedback_id = request.form.get("feedback_id", type=int)
    row = get_feedback(feedback_id) if feedback_id is not None else None
    meta = (row or {}).get("item_meta") or {}

    if meta.get("subject") and meta.get("relation") and meta.get("object"):
        driver = get_driver()
        try:
            delete_relationship(
                driver,
                subject=meta["subject"],
                relation=meta["relation"],
                obj=meta["object"],
                source=meta.get("graph_source", ""),
            )
        finally:
            driver.close()

    mark_resolved(feedback_id)
    return redirect(url_for("cancer_review"))


@app.route("/cancer/get", methods=["POST"])
def cancer_chat():
    user_input = request.form["msg"].strip()
    result = handle_chat(
        user_input,
        cancer_docsearch,
        cancer_qa_chain,
        is_cancer_red_flag,
        CANCER_EMERGENCY_RESPONSE,
        CANCER_LOW_CONFIDENCE_RESPONSE,
        graph_provider=cancer_graph,
        tools=[search_clinical_trials],
        system_prompt_text=cancer_system_prompt,
    )
    return jsonify(result)


if __name__ == "__main__":
    # debug=False: the reloader would re-exec this whole module (which loads
    # the embedding + LLM clients at import time) in a second watcher process,
    # doubling memory use and causing OOM crashes on constrained machines.
    app.run(host="0.0.0.0", port=8080, debug=False)
