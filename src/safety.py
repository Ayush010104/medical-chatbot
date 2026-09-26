"""Red-flag escalation and canned safety responses.

This module is intentionally free of any LLM calls: escalation must be
deterministic so a language model can never talk its way past it.
"""

RED_FLAG_PATTERNS = [
    "suicidal",
    "suicide",
    "kill myself",
    "want to die",
    "chest pain",
    "crushing pain in my chest",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "won't stop bleeding",
    "overdose",
    "took too many pills",
    "face drooping",
    "slurred speech",
    "sudden numbness",
    "unconscious",
    "not responding",
    "anaphylaxis",
    "severe allergic reaction",
    "bleeding during pregnancy",
    "pregnant and bleeding",
]

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call your local "
    "emergency number (911 in the US) or go to the nearest emergency room right "
    "away. I'm not able to safely help with this here."
)

LOW_CONFIDENCE_RESPONSE = (
    "I don't have reliable information to answer that confidently based on the "
    "documents I have access to. Please consult a licensed healthcare provider."
)


def is_red_flag(text: str) -> bool:
    """True if the user's message matches a known medical-emergency pattern."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in RED_FLAG_PATTERNS)
