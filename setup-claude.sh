#!/usr/bin/env bash
# setup-claude.sh — One-shot idempotent AgentMem installer for Claude Code.
# Fixes launchd plist, reloads service, installs hooks, patches settings.json.
set -eu

AGENTMEM_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/ai.agent.memory.plist"
HOOKS_DIR="$AGENTMEM_DIR/claude-code/hooks"
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
SERVICE_URL="http://localhost:18800"
LABEL="ai.agent.memory"

echo "=== AgentMem Claude Code Setup ==="
echo "Dir: $AGENTMEM_DIR"
echo ""

# ── 1. Fix launchd plist ─────────────────────────────────────────────────────
echo "[1/6] Checking launchd plist..."
if grep -q "claw-memory" "$PLIST" 2>/dev/null; then
    echo "      Fixing stale claw-memory paths → agentmem..."
    python3 -c "
import re, sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
content = content.replace('/Users/jace/code/claw-memory/start.sh',
                          '/Users/jace/code/agentmem/start.sh')
content = content.replace('<string>/Users/jace/code/claw-memory</string>',
                          '<string>/Users/jace/code/agentmem</string>')
with open(path, 'w') as f:
    f.write(content)
print('      Plist updated.')
" "$PLIST"
else
    echo "      Plist already correct."
fi

# ── 2. Reload launchd service ────────────────────────────────────────────────
echo "[2/6] Reloading launchd service..."
launchctl unload "$PLIST" 2>/dev/null || true
sleep 1
launchctl load "$PLIST"
echo "      Service loaded."

# ── 3. Wait for service health ───────────────────────────────────────────────
echo "[3/6] Waiting for service health (up to 15s)..."
HEALTHY=0
for i in $(seq 1 15); do
    STATUS=$(curl -sf -m 2 "$SERVICE_URL/health" 2>/dev/null || echo "")
    if echo "$STATUS" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
        echo "      Service healthy after ${i}s."
        HEALTHY=1
        break
    fi
    echo "      Attempt $i: not ready yet..."
    sleep 1
done
if [ $HEALTHY -eq 0 ]; then
    echo "      WARNING: Service did not become healthy within 15s."
    echo "      Check logs: tail -f ~/.openclaw/logs/memory-stderr.log"
fi

# ── 4. Create hooks directory ────────────────────────────────────────────────
echo "[4/6] Creating hooks directory..."
mkdir -p "$HOOKS_DIR"
echo "      $HOOKS_DIR"

# ── 5. Write hook scripts ────────────────────────────────────────────────────
echo "[5/6] Installing hook scripts..."

# recall.sh (UserPromptSubmit)
cat > "$HOOKS_DIR/recall.sh" << 'HOOKEOF'
#!/usr/bin/env bash
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
PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({'query': sys.argv[1], 'session_id': sys.argv[2],
                  'memory_limit_number': 6, 'include_tools': True}))
" "$PROMPT" "$SESSION" 2>/dev/null)
RESULT=$(curl -sf -m 5 -X POST http://localhost:18800/recall \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null)
python3 -c "
import json, sys
result = json.loads(sys.argv[1]) if sys.argv[1] else {}
ctx = result.get('prependContext') or ''
if ctx:
    print(json.dumps({'additionalContext': ctx}))
" "$RESULT" 2>/dev/null
exit 0
HOOKEOF

# store.sh (Stop)
cat > "$HOOKS_DIR/store.sh" << 'HOOKEOF'
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
        except: pass
if not messages: sys.exit(0)
payload = json.dumps({"messages": messages, "session_id": session_id}).encode()
req = urllib.request.Request("http://localhost:18800/store",
    data=payload, headers={"Content-Type": "application/json"}, method="POST")
try: urllib.request.urlopen(req, timeout=3)
except: pass
req2 = urllib.request.Request("http://localhost:18800/session/compress",
    data=json.dumps({"session_id": session_id, "wait": False}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
try: urllib.request.urlopen(req2, timeout=2)
except: pass
PYEOF
exit 0
HOOKEOF

# register-env.sh (SessionStart)
cat > "$HOOKS_DIR/register-env.sh" << 'HOOKEOF'
#!/usr/bin/env bash
unset PYTHONWARNINGS  # prevent invalid startup-time warning filter from spamming stderr
python3 - << 'PYEOF'
import json, subprocess, os, platform, urllib.request
def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except: return ""
env = {
    "os": platform.system(),
    "os_version": platform.mac_ver()[0] or platform.version(),
    "shell": os.environ.get("SHELL", ""),
    "cwd": os.getcwd(),
    "git_repo": sh("git rev-parse --show-toplevel"),
    "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
    "runtime": f"python{platform.python_version()}",
    "agent_model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
}
payload = json.dumps(env).encode()
req = urllib.request.Request("http://localhost:18800/register-env",
    data=payload, headers={"Content-Type": "application/json"}, method="POST")
try: urllib.request.urlopen(req, timeout=3)
except: pass
PYEOF
exit 0
HOOKEOF

# compact.sh (PostToolUse — mid-session compact, v0.9.3)
cat > "$HOOKS_DIR/compact.sh" << 'HOOKEOF'
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
HOOKEOF

chmod +x "$HOOKS_DIR/recall.sh" "$HOOKS_DIR/store.sh" "$HOOKS_DIR/register-env.sh" "$HOOKS_DIR/compact.sh"
echo "      Hook scripts written and made executable."

# ── 6. Patch ~/.claude/settings.json ────────────────────────────────────────
echo "[6/6] Patching ~/.claude/settings.json with hooks block..."
python3 - "$CLAUDE_SETTINGS" "$HOOKS_DIR" << 'PYEOF'
import json, sys

settings_path = sys.argv[1]
hooks_dir = sys.argv[2]

with open(settings_path) as f:
    settings = json.load(f)

settings["hooks"] = {
    "SessionStart": [{
        "matcher": "startup|clear|compact",
        "hooks": [{"type": "command", "command": f"{hooks_dir}/register-env.sh", "timeout": 10}]
    }],
    "UserPromptSubmit": [{
        "hooks": [{"type": "command", "command": f"{hooks_dir}/recall.sh", "timeout": 10}]
    }],
    "PostToolUse": [{
        "hooks": [{"type": "command", "command": f"{hooks_dir}/compact.sh", "timeout": 5}]
    }],
    "Stop": [{
        "hooks": [{"type": "command", "command": f"{hooks_dir}/store.sh", "timeout": 30}]
    }]
}

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2)

print(f"      Hooks block written with keys: {list(settings['hooks'].keys())}")
PYEOF

# ── Verification report ──────────────────────────────────────────────────────
echo ""
echo "=== Verification Report ==="

echo ""
echo "Service health:"
curl -sf -m 3 "$SERVICE_URL/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (service not responding)"

echo ""
echo "Memory counts:"
curl -sf -m 3 "$SERVICE_URL/stats" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for k, v in d.items():
        print(f'  {k}: {v}')
except:
    print('  (stats endpoint not available)')
" 2>/dev/null

echo ""
echo "Hooks registered in settings.json:"
python3 -c "
import json
s = json.load(open('$CLAUDE_SETTINGS'))
keys = list(s.get('hooks', {}).keys())
print(f'  {keys}')
"

echo ""
echo "launchd status:"
launchctl list | grep "$LABEL" || echo "  (not found — check launchctl list)"

echo ""
echo "Hook scripts:"
ls -la "$HOOKS_DIR/"

echo ""
echo "=== Setup complete! ==="
echo "Open a new Claude Code session to activate hooks."
