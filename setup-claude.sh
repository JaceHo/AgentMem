#!/usr/bin/env bash
# setup-claude.sh — One-shot idempotent AgentMem installer for Claude Code.
# Fixes launchd plist, reloads service, makes hooks executable, patches settings.json.
#
# Hooks live in claude-code/hooks/ — edit them there, not here.
set -eu

AGENTMEM_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/ai.agent.memory.plist"
HOOKS_DIR="$AGENTMEM_DIR/claude-code/hooks"
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
SERVICE_URL="http://localhost:18800"
LABEL="ai.agent.memory"

echo "=== AgentMem v1.0 Claude Code Setup ==="
echo "Dir: $AGENTMEM_DIR"
echo ""

# ── 1. Fix launchd plist ─────────────────────────────────────────────────────
echo "[1/6] Checking launchd plist..."
if grep -q "claw-memory" "$PLIST" 2>/dev/null; then
    echo "      Fixing stale claw-memory paths → agentmem..."
    python3 -c "
import sys
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

# ── 4. Verify hooks directory ────────────────────────────────────────────────
echo "[4/6] Verifying hooks directory..."
REQUIRED_HOOKS="recall.sh store.sh register-env.sh register-tools.sh compact.sh"
MISSING=""
for h in $REQUIRED_HOOKS; do
    if [ ! -f "$HOOKS_DIR/$h" ]; then
        MISSING="$MISSING $h"
    fi
done
if [ -n "$MISSING" ]; then
    echo "      ERROR: Missing hook scripts:$MISSING"
    echo "      Expected at $HOOKS_DIR/"
    exit 1
fi
echo "      All 5 hooks present: $HOOKS_DIR/"

# ── 5. Make hooks executable ─────────────────────────────────────────────────
echo "[5/6] Making hooks executable..."
chmod +x "$HOOKS_DIR/recall.sh" \
         "$HOOKS_DIR/store.sh" \
         "$HOOKS_DIR/register-env.sh" \
         "$HOOKS_DIR/register-tools.sh" \
         "$HOOKS_DIR/compact.sh"
echo "      chmod +x applied to all 5 hooks."

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
        "hooks": [
            {"type": "command", "command": f"{hooks_dir}/register-env.sh",   "timeout": 10},
            {"type": "command", "command": f"{hooks_dir}/register-tools.sh", "timeout": 10},
        ]
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

print(f"      SessionStart: register-env.sh + register-tools.sh")
print(f"      UserPromptSubmit: recall.sh")
print(f"      PostToolUse: compact.sh")
print(f"      Stop: store.sh")
PYEOF

# ── Verification report ──────────────────────────────────────────────────────
echo ""
echo "=== Verification Report ==="

echo ""
echo "Service health:"
curl -sf -m 3 "$SERVICE_URL/health" 2>/dev/null | python3 -m json.tool 2>/dev/null \
    || echo "  (service not responding)"

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
for event, configs in s.get('hooks', {}).items():
    for cfg in configs:
        for h in cfg.get('hooks', []):
            print(f'  {event}: {h[\"command\"].split(\"/\")[-1]}')
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
