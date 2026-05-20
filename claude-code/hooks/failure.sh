#!/usr/bin/env bash
# Hook: PostToolUseFailure — record tool failure for ToolMem reliability tracking.
# Fires when a tool use produces an error. Complements compact.sh (PostToolUse success).
# Fire-and-forget: never blocks Claude Code.
unset PYTHONWARNINGS
INPUT=$(cat)
SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")
if [ -z "$SESSION" ]; then exit 0; fi

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
error = str(d.get('error', '') or d.get('tool_response', '') or '')[:200]
print(json.dumps({
  'tool_name': tool_name,
  'success': False,
  'error': error,
  'session_id': sys.argv[2],
}))
" "$INPUT" "$SESSION" 2>/dev/null)
  curl -sf -m 2 -X POST http://localhost:18800/tool-feedback \
    -H 'Content-Type: application/json' -d "$FEEDBACK" > /dev/null 2>&1 &
fi

exit 0
