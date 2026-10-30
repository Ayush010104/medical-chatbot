"""ClinicalTrials.gov API v2 lookup — a real, live tool the cancer agent can call.

Unlike the PDQ text and the extracted graph, this is not static: it reflects
whatever is actually recruiting right now, which is exactly the kind of
question a static knowledge base structurally cannot answer well.
"""

from typing import List

import requests
from langchain_core.tools import tool

API_URL = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "Condition",
    "BriefSummary",
    "LocationCity",
    "LocationState",
    "LocationCountry",
]


def _summarize_locations(locations: List[dict], max_shown: int = 3) -> str:
    if not locations:
        return "location not specified"
    seen = []
    for loc in locations:
        label = ", ".join(
            part for part in (loc.get("city"), loc.get("state"), loc.get("country")) if part
        )
        if label and label not in seen:
            seen.append(label)
        if len(seen) >= max_shown:
            break
    suffix = f" (+{len(locations) - len(seen)} more sites)" if len(locations) > len(seen) else ""
    return "; ".join(seen) + suffix


@tool
def search_clinical_trials(condition: str, status: str = "RECRUITING", max_results: int = 5) -> str:
    """Search ClinicalTrials.gov for trials matching a cancer type or biomarker.

    Use this when the user asks about available, recruiting, or ongoing
    clinical trials — questions PDQ's static text cannot answer, since trial
    recruitment status changes continuously. Do not use it for questions about
    standard-of-care treatment, staging, or risk factors; the retrieved PDQ
    context already covers those.

    Args:
        condition: The condition to search for, e.g. "HER2-positive breast cancer",
            "non-small cell lung cancer", "BRCA1 mutation breast cancer".
        status: One of RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or ALL.
            Defaults to RECRUITING (what a clinician usually means by "available trials").
        max_results: How many trials to return (1-10, default 5).
    """
    max_results = max(1, min(max_results, 10))
    params = {
        "query.cond": condition,
        "pageSize": max_results,
        "fields": ",".join(FIELDS),
    }
    if status != "ALL":
        params["filter.overallStatus"] = status

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"ClinicalTrials.gov lookup failed: {exc}"

    studies = response.json().get("studies", [])
    if not studies:
        return f"No trials found on ClinicalTrials.gov for '{condition}' with status {status}."

    lines = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        nct_id = protocol.get("identificationModule", {}).get("nctId", "unknown")
        title = protocol.get("identificationModule", {}).get("briefTitle", "untitled")
        overall_status = protocol.get("statusModule", {}).get("overallStatus", "unknown")
        phases = protocol.get("designModule", {}).get("phases", []) or ["not specified"]
        summary = protocol.get("descriptionModule", {}).get("briefSummary", "").strip()
        locations = protocol.get("contactsLocationsModule", {}).get("locations", [])

        lines.append(
            f"{nct_id} ({overall_status}, {'/'.join(phases)}): {title}\n"
            f"  Locations: {_summarize_locations(locations)}\n"
            f"  Summary: {summary[:300]}{'...' if len(summary) > 300 else ''}\n"
            f"  Link: https://clinicaltrials.gov/study/{nct_id}"
        )

    return "\n\n".join(lines)
