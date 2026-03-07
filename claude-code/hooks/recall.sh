#!/usr/bin/env bash
# Hook: UserPromptSubmit — injects memory context before every user prompt.
# Reads stdin JSON from Claude Code, calls /recall, outputs additionalContext.
unset PYTHONWARNINGS  # prevent invalid startup-time warning filter from spamming stderr
INPUT=$(cat)

PROMPT=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('prompt', ''))
" "$INPUT" 2>/dev/null || echo "")

SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")

# Build payload and call AgentMem /recall (5s timeout, silent fail)
PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'query': sys.argv[1],
    'session_id': sys.argv[2],
    'memory_limit_number': 6,
    'include_tools': True
}))
" "$PROMPT" "$SESSION" 2>/dev/null)

RESULT=$(curl -sf -m 5 -X POST http://localhost:18800/recall \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null)

# Extract prependContext and emit as additionalContext for Claude Code
python3 -c "
import json, sys
result = json.loads(sys.argv[1]) if sys.argv[1] else {}
ctx = result.get('prependContext') or ''
if ctx:
    print(json.dumps({'additionalContext': ctx}))
# else: no output = no injection, Claude Code continues normally
" "$RESULT" 2>/dev/null

exit 0
