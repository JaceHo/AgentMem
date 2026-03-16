#!/bin/bash
# dev.sh — restart agentmem launchd service and re-register skills/agents
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="ai.agent.memory"
PLIST_SRC="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${AGENTMEM_PORT:-18800}"

# Parse --force flag
FORCE=false
for arg in "$@"; do [ "$arg" = "--force" ] && FORCE=true; done

# Check for active connections
check_connections() {
    local conns
    conns=$(lsof -i "TCP:$1" -s TCP:ESTABLISHED -n -P 2>/dev/null | tail -n +2 || true)
    [ -z "$conns" ] && return 0
    local count; count=$(echo "$conns" | wc -l | tr -d ' ')
    echo "WARNING: $count active connection(s) on :$1"
    echo "$conns" | awk '{printf "  %-12s pid=%-6s %s\n", $1, $2, $NF}'
    return 1
}

if ! check_connections "$PORT"; then
    if [ "$FORCE" = false ]; then
        echo ""
        echo "restart aborted — active connections will be dropped."
        echo "use --force to restart anyway."
        exit 1
    fi
    echo "WARNING: proceeding with --force, dropping active connections..."
fi

uid="$(id -u)"
if launchctl print "gui/$uid/$LABEL" &>/dev/null 2>&1; then
    echo "restarting $LABEL via launchd..."
    launchctl kickstart -k "gui/$uid/$LABEL"
else
    echo "service not loaded — bootstrapping..."
    launchctl bootstrap "gui/$uid" "$PLIST_SRC"
fi

# AgentMem loads MiniLM embedding model at startup (~30s) — poll until ready
echo "waiting for agentmem (MiniLM warmup may take ~30s)..."
for i in $(seq 1 80); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "up on :$PORT"
        break
    fi
    printf "."
    sleep 0.5
done
echo ""

# Re-register skills + agents now that service is ready
echo "registering skills + agents..."
python3 "$SCRIPT_DIR/register-skills.py"

echo ""
echo "--- stdout (last 20 lines) ---"
tail -20 "$HOME/.openclaw/logs/memory-stdout.log" 2>/dev/null || true
echo ""
echo "--- stderr (last 10 lines) ---"
tail -10 "$HOME/.openclaw/logs/memory-stderr.log" 2>/dev/null || true
