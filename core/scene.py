"""
Scene detection: language + domain from raw text.
Used to tag memories at store time and filter them at recall time.
"""

import re

_ZH = re.compile(r'[\u4e00-\u9fff]')
_JA = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')

_DOMAINS: dict[str, list[str]] = {
    "trading":    ["股票","交易","a股","行情","买入","卖出","涨停","跌停","接口","华宝","证券",
                   "trade","stock","market","portfolio","equity","ticker","forex"],
    "coding":     ["python","javascript","typescript","golang","rust","java","swift",
                   "function","class","bug","api","backend","frontend","refactor","test",
                   "代码","函数","接口","调试","bug","编程"],
    "devops":     ["docker","redis","nginx","systemd","ssh","port","tunnel","deploy",
                   "kubernetes","server","cron","launchd","plist","部署","服务器"],
    "ai":         ["model","llm","embedding","vector","prompt","agent","rag","claude",
                   "gpt","minimax","glm","openai","fine-tune","inference","模型","向量","嵌入"],
    "finance":    ["revenue","profit","roi","invest","fund","vc","startup","raise",
                   "收入","利润","投资","融资","估值"],
    "capability": ["tool","skill","plugin","mcp","resource","ability","capability",
                   "what can you","what tools","do you have","can you use","help me with",
                   "工具","技能","插件","能力","你能","你会","有什么功能"],
}

# Capability query patterns (separate from domain — these trigger tool recall)
_CAPABILITY_QUERY = re.compile(
    r'\b(what\s+(can|tools?|skills?|plugin|ability|abilities|capabilities?|commands?|resources?)'
    r'|do\s+you\s+have\s+a?\s*(tool|skill|way|command|ability)'
    r'|can\s+you\s+(use|run|execute|search|read|write|fetch|access)'
    r'|how\s+do\s+i\s+(use|run|execute)'
    r'|list\s+(all\s+)?(tools?|skills?|commands?|capabilities?)'
    r'|你(能|会|有)(什么|哪些|工具|技能|命令|插件)'
    r'|有什么(工具|技能|功能|命令)'
    r')\b',
    re.I
)


def detect_language(text: str) -> str:
    zh = len(_ZH.findall(text))
    ja = len(_JA.findall(text))
    total = max(len(text), 1)
    if zh / total > 0.12:
        return "zh"
    if ja / total > 0.12:
        return "ja"
    return "en"


def detect_domain(text: str) -> str:
    low = text.lower()
    best, best_score = "general", 0
    for domain, kws in _DOMAINS.items():
        score = sum(1 for kw in kws if kw in low)
        if score > best_score:
            best, best_score = domain, score
    return best


def is_capability_query(text: str) -> bool:
    """Returns True if the query is asking about agent tools/abilities/capabilities."""
    return bool(_CAPABILITY_QUERY.search(text))


def detect(text: str) -> dict[str, str]:
    return {
        "language":           detect_language(text),
        "domain":             detect_domain(text),
        "is_capability_query": is_capability_query(text),
    }
