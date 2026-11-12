"""
Ethical guard-rails required by the product spec:
no facial identification, no biometric matching, no automated decisions
about people. Requests that ask for these are blocked *before* they reach
the model, and the analyst sees a clear explanation.
"""

import re
from typing import Optional

# Patterns that indicate identity / biometric / judgement requests.
_BLOCKED_PATTERNS = [
    r"\bwho\s+(is|are)\s+(this|that|these|those|the)\s+(person|man|woman|people|guy|girl|boy|individual|suspect|driver|owner|victim)",
    r"\b(identify|recogni[sz]e|name)\s+(this|that|the|these)?\s*(person|people|man|woman|face|faces|individual|suspect)",
    r"\bface\s*(recognition|match|matching|id)\b",
    r"\b(match|compare)\s+(the\s+)?faces?\b",
    r"\bsame\s+person\b",
    r"\bbiometric",
    r"\b(is|are)\s+(he|she|they|this person)\s+(guilty|a criminal|lying|dangerous)\b",
    r"\bwhat\s+is\s+(his|her|their)\s+name\b",
    r"\b(ethnicity|race|religion)\s+of\b",
]
_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

POLICY_MESSAGE = (
    "This request was not sent to the model. The Visual Analytics Suite does not perform "
    "facial identification, biometric matching, or judgements about individuals (identity, "
    "guilt, ethnicity, etc.). You can instead ask about **visible attributes** — for example "
    "clothing, items carried, posture, or the number of people."
)

# Added to every system prompt so the model follows the same rules.
SAFETY_SYSTEM_RULES = (
    "Never identify real people by name, never guess identity, ethnicity, religion or other "
    "sensitive attributes, and never judge guilt or intent. Describe only visible attributes."
)


def check_request(text: str) -> Optional[str]:
    """Return a policy message if the request is not allowed, else None."""
    if not text:
        return None
    for pattern in _BLOCKED_RE:
        if pattern.search(text):
            return POLICY_MESSAGE
    return None
