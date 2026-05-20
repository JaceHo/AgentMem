#!/usr/bin/env bash
# Hook: PreCompact — re-inject memory before context compaction.
# Fires when Claude Code is about to compact the context window.
# Re-injects cross-session memory so the compacted summary retains
# facts, persona, and session context that would otherwise be lost.
unset PYTHONWARNINGS

if ! nc -z -G 1 localhost 18800 2>/dev/null; then
  exit 0
fi

INPUT=$(cat)
SESSION=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print(d.get('session_id', ''))
" "$INPUT" 2>/dev/null || echo "")

# Use the session summary as the query since we want broad context for compaction
PAYLOAD=$(python3 -c "
import json, sys
session_id = sys.argv[1]
print(json.dumps({
  'query': 'session context summary preferences rules decisions',
  'session_id': session_id,
  'memory_limit_number': 8,
  'include_tools': False,
  'include_procedures': True,
  'token_budget': 1200,
}))
" "$SESSION" 2>/dev/null)

RESULT=$(curl -sf -m 3 -X POST http://localhost:18800/recall \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null)

python3 -c "
import json, sys
result = json.loads(sys.argv[1]) if sys.argv[1] else {}
ctx = result.get('prependContext') or ''
if ctx:
    print(json.dumps({'additionalContext': ctx}))
" "$RESULT" 2>/dev/null

exit 0
