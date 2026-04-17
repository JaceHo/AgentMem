#!/usr/bin/env bash
# AgentMem recall hook — fast & non-blocking.
# Exits silently in <1s if AgentMem is unreachable, so Claude Code never stalls.
unset PYTHONWARNINGS

# Fast liveness probe: skip everything if port 18800 isn't accepting connections.
if ! nc -z -G 1 localhost 18800 2>/dev/null; then
  exit 0
fi

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
PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({'query': sys.argv[1], 'session_id': sys.argv[2],
                  'memory_limit_number': 6, 'include_tools': True,
                  'include_procedures': True}))
" "$PROMPT" "$SESSION" 2>/dev/null)

# Hard-cap recall to 2s. If it can't answer that fast, skip silently.
RESULT=$(curl -sf -m 2 -X POST http://localhost:18800/recall \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null)

python3 -c "
import json, sys
result = json.loads(sys.argv[1]) if sys.argv[1] else {}
ctx = result.get('prependContext') or ''
if ctx:
    print(json.dumps({'additionalContext': ctx}))
" "$RESULT" 2>/dev/null
exit 0
