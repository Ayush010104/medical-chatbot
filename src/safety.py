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


# Cancer department: the audience is oncology clinicians discussing patient
# symptoms clinically ("manage dyspnea from malignant pleural effusion"), so
# the general ER-trigger patterns above (chest pain, can't breathe, etc.)
# would nonsensically block ordinary clinical questions. Self-harm content is
# kept as a universal safety net regardless of who's typing.
CANCER_RED_FLAG_PATTERNS = [
    "suicidal",
    "suicide",
    "kill myself",
    "want to die",
]

CANCER_EMERGENCY_RESPONSE = (
    "This message suggests a possible mental health crisis. If you or someone "
    "else is in immediate danger, call your local emergency number (911 in "
    "the US) or the 988 Suicide & Crisis Lifeline. I'm not able to safely "
    "help with this here."
)

CANCER_LOW_CONFIDENCE_RESPONSE = (
    "The indexed PDQ summaries don't contain enough directly relevant "
    "information to answer this confidently. Recommend correlating with the "
    "full current guideline text, primary literature, or tumor board "
    "discussion rather than relying on this response."
)


def is_cancer_red_flag(text: str) -> bool:
    """True if the message matches the cancer department's (narrower) escalation list."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in CANCER_RED_FLAG_PATTERNS)
