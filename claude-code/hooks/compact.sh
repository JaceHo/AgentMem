#!/usr/bin/env bash
# Hook: PostToolUse — mid-session compact (v0.9.3, claude-mem Endless Mode inspired).
# Fires after every tool use; calls /session/compact only when session KV > 3000 chars.
# Fire-and-forget: we do NOT wait for the response so tool latency is unaffected.
unset PYTHONWARNINGS
INPUT=$(cat)
SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")
if [ -z "$SESSION" ]; then exit 0; fi
PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({'session_id': sys.argv[1], 'threshold_chars': 3000}))
" "$SESSION" 2>/dev/null)
curl -sf -m 3 -X POST http://localhost:18800/session/compact \
  -H 'Content-Type: application/json' -d "$PAYLOAD" > /dev/null 2>&1 &
exit 0
