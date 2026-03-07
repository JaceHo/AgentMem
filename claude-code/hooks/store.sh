#!/usr/bin/env bash
# Hook: Stop — persists conversation transcript to long-term memory on session end.
unset PYTHONWARNINGS  # prevent invalid startup-time warning filter from spamming stderr
INPUT=$(cat)

SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")

TRANSCRIPT=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('transcript_path', ''))
" "$INPUT" 2>/dev/null || echo "")

if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then exit 0; fi

# Parse JSONL transcript → messages → call /store (fire-and-forget)
# Then promote Tier 1 session KV → Tier 2 long-term via /session/compress
python3 - "$SESSION" "$TRANSCRIPT" << 'PYEOF'
import json, sys, urllib.request

session_id = sys.argv[1]
transcript_path = sys.argv[2]

messages = []
with open(transcript_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            role = entry.get("role") or entry.get("type", "")
            content = entry.get("content") or entry.get("message", "")
            if role and content:
                messages.append({"role": role, "content": str(content)})
        except:
            pass

if not messages:
    sys.exit(0)

# Store messages
payload = json.dumps({"messages": messages, "session_id": session_id}).encode()
req = urllib.request.Request(
    "http://localhost:18800/store",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    urllib.request.urlopen(req, timeout=3)
except:
    pass

# Promote Tier 1 → Tier 2 (async, non-blocking)
req2 = urllib.request.Request(
    "http://localhost:18800/session/compress",
    data=json.dumps({"session_id": session_id, "wait": False}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    urllib.request.urlopen(req2, timeout=2)
except:
    pass
PYEOF

exit 0
