#!/usr/bin/env bash
# Hook: PostToolUse — mid-session compact + ToolMem feedback (v0.9.5).
# Fires after every tool use:
#   1. /session/compact — trim Tier 1 KV if > 3000 chars (claude-mem Endless Mode)
#   2. /tool-feedback   — record success/fail for ToolMem capability tracking
# Fire-and-forget: we do NOT wait for responses so tool latency is unaffected.
unset PYTHONWARNINGS
INPUT=$(cat)
SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")
if [ -z "$SESSION" ]; then exit 0; fi

# 1. Mid-session compact
PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({'session_id': sys.argv[1], 'threshold_chars': 3000}))
" "$SESSION" 2>/dev/null)
curl -sf -m 3 -X POST http://localhost:18800/session/compact \
  -H 'Content-Type: application/json' -d "$PAYLOAD" > /dev/null 2>&1 &

# 2. ToolMem feedback: extract tool_name + heuristic success, fire-and-forget
TOOL_NAME=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('tool_name', ''))
" "$INPUT" 2>/dev/null || echo "")
if [ -n "$TOOL_NAME" ]; then
  FEEDBACK=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
tool_name = d.get('tool_name', '')
output = str(d.get('tool_response', '') or d.get('output', '') or d.get('tool_result', '') or '')
error_words = ['error', 'exception', 'traceback', 'failed', 'not found', 'permission denied', 'timeout', 'command not found']
success = not any(w in output.lower() for w in error_words)
print(json.dumps({'tool_name': tool_name, 'success': success, 'session_id': sys.argv[2]}))
" "$INPUT" "$SESSION" 2>/dev/null)
  curl -sf -m 2 -X POST http://localhost:18800/tool-feedback \
    -H 'Content-Type: application/json' -d "$FEEDBACK" > /dev/null 2>&1 &
fi

exit 0
