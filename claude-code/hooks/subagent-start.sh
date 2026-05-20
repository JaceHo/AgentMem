#!/usr/bin/env bash
# Hook: SubagentStart — inject cross-session memory into subagent spawn.
# Fires when Claude Code launches a subagent (Agent tool call).
# Returns additionalContext so the subagent starts with relevant memory,
# not cold. Mirrors recall.sh but targets the subagent session.
unset PYTHONWARNINGS

if ! nc -z -G 1 localhost 18800 2>/dev/null; then
  exit 0
fi

INPUT=$(cat)
SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
# SubagentStart provides session_id (parent) or subagent_id
print(d.get('session_id', '') or d.get('subagent_id', ''))
" "$INPUT" 2>/dev/null || echo "")

PAYLOAD=$(python3 -c "
import json, sys
session_id = sys.argv[1]
print(json.dumps({
  'query': 'session context rules preferences decisions procedures',
  'session_id': session_id,
  'memory_limit_number': 6,
  'include_tools': True,
  'include_procedures': True,
  'token_budget': 900,
  'enable_hyde': False,
}))
" "$SESSION" 2>/dev/null)

RESULT=$(curl -sf -m 4 -X POST http://localhost:18800/recall \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null)

python3 -c "
import json, sys
result = json.loads(sys.argv[1]) if sys.argv[1] else {}
ctx = result.get('prependContext') or ''
if ctx:
    print(json.dumps({'additionalContext': ctx}))
" "$RESULT" 2>/dev/null
exit 0
