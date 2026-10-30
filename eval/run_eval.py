"""Ablation harness for the Cancer Department: vector-only vs vector+graph vs
vector+graph+tools, on a hand-curated, source-grounded QA set.

Reuses the exact production wiring from app.py (same docsearch, same graph
provider, same prompt, same confidence threshold) rather than reimplementing
anything — this measures the real system, not a simplified stand-in.

Correctness is checked programmatically (substring presence / confidence
field), not via an LLM judge — keeps this cheap and avoids introducing a
second model's biases into the evaluation of the first.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    CANCER_LOW_CONFIDENCE_RESPONSE,
    cancer_docsearch,
    cancer_graph,
    cancer_qa_chain,
    cancer_system_prompt,
    handle_chat,
    is_cancer_red_flag,
    search_clinical_trials,
)
from src.safety import CANCER_EMERGENCY_RESPONSE  # noqa: E402

QA_SET_PATH = os.path.join(os.path.dirname(__file__), "qa_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

CONDITIONS = ["vector_only", "vector_graph", "production"]


def run_condition(condition: str, question: str) -> dict:
    kwargs = dict(
        docsearch=cancer_docsearch,
        qa_chain=cancer_qa_chain,
        is_flag_fn=is_cancer_red_flag,
        emergency_response=CANCER_EMERGENCY_RESPONSE,
        low_confidence_response=CANCER_LOW_CONFIDENCE_RESPONSE,
    )
    if condition == "vector_graph":
        kwargs["graph_provider"] = cancer_graph
    elif condition == "production":
        kwargs["graph_provider"] = cancer_graph
        kwargs["tools"] = [search_clinical_trials]
        kwargs["system_prompt_text"] = cancer_system_prompt

    return handle_chat(question, **kwargs)


def score(item: dict, result: dict) -> dict:
    answer = (result.get("answer") or "").lower()
    confidence = result.get("confidence")
    sources = result.get("sources") or []

    correctness = None
    expected = item.get("expected_contains_any") or []
    if expected:
        correctness = any(kw.lower() in answer for kw in expected)

    refusal_correct = None
    if item.get("expects_refusal"):
        refusal_correct = confidence in ("low", "emergency")

    tool_call_correct = None
    if "expects_tool_call" in item:
        tool_called = any(s.get("content_type") == "clinical_trial" for s in sources)
        tool_call_correct = tool_called == item["expects_tool_call"]

    return {
        "confidence": confidence,
        "correctness": correctness,
        "refusal_correct": refusal_correct,
        "tool_call_correct": tool_call_correct,
        "n_vector_sources": sum(1 for s in sources if s.get("content_type") in ("text", "table")),
        "n_graph_facts": sum(1 for s in sources if s.get("content_type") == "graph_fact"),
        "n_trial_results": sum(1 for s in sources if s.get("content_type") == "clinical_trial"),
        "answer": result.get("answer"),
    }


def main() -> None:
    with open(QA_SET_PATH, encoding="utf-8") as f:
        qa_set = json.load(f)

    results = []
    total_calls = len(qa_set) * len(CONDITIONS)
    done = 0
    start = time.time()

    for item in qa_set:
        row = {"id": item["id"], "category": item["category"], "question": item["question"]}
        for condition in CONDITIONS:
            result = run_condition(condition, item["question"])
            row[condition] = score(item, result)
            done += 1
            print(f"  [{done}/{total_calls}] {item['id']} / {condition} "
                  f"-> confidence={row[condition]['confidence']}, correct={row[condition]['correctness']}")
        results.append(row)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
