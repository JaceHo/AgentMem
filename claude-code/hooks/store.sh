#!/usr/bin/env bash
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
python3 - "$SESSION" "$TRANSCRIPT" << 'PYEOF'
import json, sys, urllib.request
session_id = sys.argv[1]
transcript_path = sys.argv[2]
messages = []
tool_sequence = []  # AutoTool TIG: ordered tool names used in session
with open(transcript_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            entry = json.loads(line)
            role = entry.get("role") or entry.get("type", "")
            content = entry.get("content") or entry.get("message", "")
            if role and content:
                messages.append({"role": role, "content": str(content)})
            # Extract tool_use blocks from assistant messages for TIG
            raw_content = content if isinstance(content, list) else \
                          (entry.get("message", {}) or {}).get("content", [])
            if isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "")
                        if name:
                            tool_sequence.append(name)
        except: pass
if not messages: sys.exit(0)
# 1. Store messages → long-term memory
payload = json.dumps({"messages": messages, "session_id": session_id}).encode()
req = urllib.request.Request("http://localhost:18800/store",
    data=payload, headers={"Content-Type": "application/json"}, method="POST")
try: urllib.request.urlopen(req, timeout=3)
except: pass
# 2. Promote session Tier 1 → Tier 2
req2 = urllib.request.Request("http://localhost:18800/session/compress",
    data=json.dumps({"session_id": session_id, "wait": False}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
try: urllib.request.urlopen(req2, timeout=2)
except: pass
# 3. Record tool sequence into TIG (AutoTool, v0.9.5)
if len(tool_sequence) >= 2:
    req3 = urllib.request.Request("http://localhost:18800/record-tool-sequence",
        data=json.dumps({"sequence": tool_sequence, "session_id": session_id}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try: urllib.request.urlopen(req3, timeout=2)
    except: pass
    # 4. AWO meta-tool detection (v0.9.6): synthesize composite procedures from TIG chains.
    #    Idempotent — knn_search dedup prevents re-adding existing meta-tool entries.
    req4 = urllib.request.Request("http://localhost:18800/tool-graph/detect-meta-tools?threshold=5",
        data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try: urllib.request.urlopen(req4, timeout=3)
    except: pass
PYEOF
exit 0
