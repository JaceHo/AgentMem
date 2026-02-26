"""Rule-based fact extractor from conversation messages."""
import re
import time
from dataclasses import dataclass, field


@dataclass
class ExtractedFact:
    content: str
    category: str = "general"
    confidence: float = 0.8


# Patterns: (regex, category)
_PATTERNS = [
    (re.compile(r"(?:my name is|I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I), "identity"),
    (re.compile(r"(?:I (?:work|am working) (?:at|for)|I work as)\s+(.+?)(?:\.|,|$)", re.I), "work"),
    (re.compile(r"(?:I (?:like|love|prefer|enjoy|hate|dislike))\s+(.+?)(?:\.|,|$)", re.I), "preference"),
    (re.compile(r"(?:I (?:live|am|stay) in|I'm from|I'm based in)\s+(.+?)(?:\.|,|$)", re.I), "location"),
    (re.compile(r"(?:my (?:phone|email|address|birthday|goal|project) is)\s+(.+?)(?:\.|,|$)", re.I), "personal"),
    (re.compile(r"(?:remember|note|important):\s*(.+?)(?:\.|$)", re.I), "reminder"),
    (re.compile(r"(?:always|never|always use|never use)\s+(.+?)(?:\.|,|$)", re.I), "rule"),
    (re.compile(r"(?:the (?:api|key|token|secret|password) (?:is|for .+ is))\s+(\S+)", re.I), "credential"),
    (re.compile(r"(?:deadline|due date|by)\s+([\w\s,]+\d{4}|\d{4}-\d{2}-\d{2})", re.I), "deadline"),
]


def extract(messages: list[dict]) -> list[ExtractedFact]:
    """Extract facts from the last user+assistant turn."""
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    # Focus on user messages (most signal)
    user_text = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
        for m in messages if m.get("role") == "user"
    )

    for pattern, category in _PATTERNS:
        for match in pattern.finditer(user_text):
            snippet = match.group(0).strip()[:200]
            if snippet and snippet not in seen:
                seen.add(snippet)
                facts.append(ExtractedFact(content=snippet, category=category))

    # Also treat long assistant replies as episodic content (not facts)
    return facts
