#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AgentMem — Local Agent Memory Service Control Script
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${AGENTMEM_PORT:-18800}"
SERVICE_LABEL="ai.agent.memory"
PLIST_DST="$HOME/Library/LaunchAgents/$SERVICE_LABEL.plist"
LOG_DIR="$HOME/.agentmem/logs"
LOG_STDOUT="$LOG_DIR/stdout.log"
LOG_STDERR="$LOG_DIR/stderr.log"
mkdir -p "$LOG_DIR"

# ── Bench constants ───────────────────────────────────────────────────────────
BENCH_PORT=18899
BENCH_DB=15
BENCH_LOG=/tmp/agentmem-bench.log
BENCH_PID=/tmp/agentmem-bench.pid

print_banner() {
    echo "════════════════════════════════════════"
    echo "  AgentMem — Local Agent Memory Service"
    echo "════════════════════════════════════════"
}

check_deps() {
    if ! command -v uv &>/dev/null; then
        echo "uv required (brew install uv)"; exit 1
    fi
}

is_running() {
    if pgrep -f "uvicorn main:app" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

_kill_stale() {
    local stale
    stale=$(lsof -ti "tcp:$PORT" 2>/dev/null || true)
    if [ -n "$stale" ]; then
        echo "  Killing stale process on :$PORT (pid $stale)"
        echo "$stale" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

_wait_ready() {
    local port="${1:-$PORT}"
    local max="${2:-80}"
    echo -n "  Waiting for service"
    for i in $(seq 1 "$max"); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        printf "."
        sleep 0.5
    done
    echo " timeout"
    return 1
}

_check_connections() {
    local conns
    conns=$(lsof -i "TCP:$PORT" -s TCP:ESTABLISHED -n -P 2>/dev/null | tail -n +2 || true)
    [ -z "$conns" ] && return 0
    local count; count=$(echo "$conns" | wc -l | tr -d ' ')
    echo "WARNING: $count active connection(s) on :$PORT"
    echo "$conns" | awk '{printf "  %-12s pid=%-6s %s\n", $1, $2, $NF}'
    return 1
}

_register_capabilities() {
    for i in $(seq 1 40); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            uv run python "$SCRIPT_DIR/scripts/register-skills.py" \
                && echo "[agentmem] skills+agents registered" \
                || echo "[agentmem] skills+agents registration failed"
            return
        fi
        sleep 1
    done
    echo "[agentmem] skills+agents registration skipped (service not ready)"
}

# ── Commands ──────────────────────────────────────────────────────────────────

case "${1:-help}" in

    start)
        print_banner
        check_deps

        if is_running; then
            echo "Service already running on :$PORT"
            exit 0
        fi

        uid="$(id -u)"
        if launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
            echo "  Starting via launchd..."
            launchctl kickstart "gui/$uid/$SERVICE_LABEL"
            echo "  Started"
        else
            echo ""
            echo "  Starting AgentMem on :$PORT (no launchd — run 'enable' for auto-start)"
            echo "  API:  http://localhost:$PORT"
            echo ""
            _kill_stale
            _register_capabilities &
            uv run uvicorn main:app \
                --host 0.0.0.0 \
                --port "$PORT" \
                --log-level info \
                --timeout-graceful-shutdown 0 &
            echo "  Started (PID: $!)"
        fi
        ;;

    start-fg)
        print_banner
        check_deps
        echo "  Starting in foreground on :$PORT"
        echo ""
        _kill_stale
        ( _register_capabilities & )  # double-fork: orphan to launchd, not zombie of uv run
        exec uv run uvicorn main:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --log-level info \
            --timeout-graceful-shutdown 0
        ;;

    stop)
        echo "Stopping AgentMem..."
        uid="$(id -u)"
        if launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
            launchctl stop "$SERVICE_LABEL"
            echo "Stopped (launchd; KeepAlive will restart — use 'disable' to prevent)"
        elif pgrep -f "uvicorn main:app" >/dev/null 2>&1; then
            pkill -f "uvicorn main:app"
            echo "Stopped"
        else
            echo "Not running"
        fi
        ;;

    restart)
        # Parse --force flag
        FORCE=false
        for arg in "$@"; do [ "$arg" = "--force" ] && FORCE=true; done

        uid="$(id -u)"
        if ! $FORCE && ! _check_connections; then
            echo ""
            echo "Restart aborted — active connections will be dropped."
            echo "Use --force to restart anyway."
            exit 1
        fi
        [ "$FORCE" = true ] && echo "WARNING: proceeding with --force..."

        if launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
            echo "Restarting via launchd (kickstart -k)..."
            launchctl kickstart -k "gui/$uid/$SERVICE_LABEL"
        else
            echo "launchd service not installed — use 'enable' first, or 'start-fg' to run manually"
            exit 1
        fi

        _wait_ready "$PORT" 80
        echo ""
        echo "Re-registering skills + agents..."
        uv run python "$SCRIPT_DIR/scripts/register-skills.py"
        echo ""
        echo "--- stdout (last 20 lines) ---"
        tail -20 "$LOG_STDOUT" 2>/dev/null || true
        echo ""
        echo "--- stderr (last 10 lines) ---"
        tail -10 "$LOG_STDERR" 2>/dev/null || true
        ;;

    status)
        if is_running; then
            echo "AgentMem is running on :$PORT"
            pgrep -f "uvicorn main:app" | while read -r p; do
                ps -o pid=,ppid=,lstart= -p "$p" 2>/dev/null \
                    | awk '{printf "  PID %-6s parent=%-6s started=%s %s %s %s %s\n", $1,$2,$3,$4,$5,$6,$7}'
            done
            echo ""
            curl -sf "http://localhost:$PORT/health" 2>/dev/null \
                | python3 -m json.tool 2>/dev/null \
                || echo "  (API not responding yet)"
            echo ""
            curl -sf "http://localhost:$PORT/stats" 2>/dev/null \
                | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for k, v in d.items(): print(f'  {k}: {v}')
except: pass
" 2>/dev/null
        else
            echo "AgentMem is not running"
        fi
        ;;

    health)
        echo "Health:"
        curl -sf "http://localhost:$PORT/health" | python3 -m json.tool 2>/dev/null \
            || echo "  (not reachable)"
        ;;

    enable)
        echo "Installing launchd service..."
        PLIST_SRC="$SCRIPT_DIR/ai.agent.memory.plist"
        if [ ! -f "$PLIST_SRC" ]; then
            echo "ERROR: plist not found at $PLIST_SRC"
            exit 1
        fi
        mkdir -p "$HOME/Library/LaunchAgents"
        mkdir -p "$(dirname "$LOG_STDOUT")"

        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null || true
        _kill_stale

        cp "$PLIST_SRC" "$PLIST_DST"
        launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

        echo "Service installed and started"
        echo "  Plist: $PLIST_DST"
        echo "  Logs:  $LOG_STDOUT"
        echo ""
        _wait_ready "$PORT" 60 || true
        ;;

    disable)
        echo "Removing launchd service..."
        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null || true
        rm -f "$PLIST_DST"
        _kill_stale
        echo "Service removed."
        ;;

    logs)
        echo "--- stdout (last 40 lines) ---"
        tail -40 "$LOG_STDOUT" 2>/dev/null || echo "  (no log at $LOG_STDOUT)"
        echo ""
        echo "--- stderr (last 20 lines) ---"
        tail -20 "$LOG_STDERR" 2>/dev/null || echo "  (no log at $LOG_STDERR)"
        ;;

    bench)
        subcmd="${2:-start}"
        case "$subcmd" in
            start)
                stale=$(lsof -ti "tcp:$BENCH_PORT" 2>/dev/null || true)
                if [ -n "$stale" ]; then
                    echo "Killing stale process on :$BENCH_PORT (pid $stale)..."
                    kill -9 "$stale" 2>/dev/null || true
                    sleep 1
                fi
                echo "Starting AgentMem benchmark service..."
                echo "  Port    : $BENCH_PORT"
                echo "  Redis DB: $BENCH_DB  (redis://localhost:6379/$BENCH_DB)"
                echo "  Log     : $BENCH_LOG"
                echo ""
                export REDIS_URL="redis://localhost:6379/$BENCH_DB"
                export AMAC_THRESHOLD="0.05"
                export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
                export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
                uv run uvicorn main:app \
                    --host 127.0.0.1 \
                    --port "$BENCH_PORT" \
                    --log-level warning \
                    > "$BENCH_LOG" 2>&1 &
                echo $! > "$BENCH_PID"
                echo "Benchmark service started (pid $!, log: $BENCH_LOG)"
                _wait_ready "$BENCH_PORT" 60 || { echo "Timeout! Check $BENCH_LOG"; exit 1; }
                echo ""
                echo "Benchmark service up at http://localhost:$BENCH_PORT"
                echo "Stop with:  $0 bench stop"
                echo "Flush DB:   $0 bench flush"
                ;;
            fg)
                echo "Starting benchmark service in foreground on :$BENCH_PORT"
                export REDIS_URL="redis://localhost:6379/$BENCH_DB"
                export AMAC_THRESHOLD="0.05"
                exec uv run uvicorn main:app \
                    --host 127.0.0.1 \
                    --port "$BENCH_PORT" \
                    --log-level info
                ;;
            stop)
                stopped=0
                if [ -f "$BENCH_PID" ]; then
                    pid=$(cat "$BENCH_PID")
                    kill "$pid" 2>/dev/null && stopped=1
                    rm -f "$BENCH_PID"
                fi
                stale=$(lsof -ti "tcp:$BENCH_PORT" 2>/dev/null || true)
                if [ -n "$stale" ]; then
                    kill -9 "$stale" 2>/dev/null && stopped=1
                fi
                [ "$stopped" -eq 1 ] && echo "Benchmark service stopped" || echo "Not running"
                ;;
            flush)
                redis-cli -n "$BENCH_DB" FLUSHDB \
                    && echo "Flushed Redis db=$BENCH_DB (benchmark data only)"
                ;;
            *)
                echo "Usage: $0 bench [start|fg|stop|flush]"
                exit 1
                ;;
        esac
        ;;

    setup)
        # Install / refresh AgentMem hooks in Claude Code settings.json
        HOOKS_DIR="$SCRIPT_DIR/claude-code/hooks"
        CLAUDE_SETTINGS="$HOME/.claude/settings.json"

        echo "=== AgentMem Claude Code Setup ==="
        echo "Dir: $SCRIPT_DIR"
        echo ""

        # Fix launchd plist paths if stale
        echo "[1/5] Checking launchd plist..."
        if [ -f "$PLIST_DST" ] && grep -q "claw-memory" "$PLIST_DST" 2>/dev/null; then
            echo "      Fixing stale claw-memory paths → agentmem..."
            python3 -c "
import sys
path = sys.argv[1]
with open(path) as f: content = f.read()
content = content.replace('/Users/jace/code/claw-memory/start.sh',
                          '/Users/jace/code/agentmem/agentmem.sh')
content = content.replace('<string>/Users/jace/code/claw-memory</string>',
                          '<string>/Users/jace/code/agentmem</string>')
with open(path, 'w') as f: f.write(content)
print('      Plist updated.')
" "$PLIST_DST"
        else
            echo "      Plist OK."
        fi

        # Reload service
        echo "[2/5] Reloading launchd service..."
        launchctl unload "$PLIST_DST" 2>/dev/null || true
        sleep 1
        launchctl load "$PLIST_DST"
        echo "      Service loaded."

        # Wait for health
        echo "[3/5] Waiting for service health (up to 40s)..."
        for i in $(seq 1 80); do
            STATUS=$(curl -sf -m 2 "http://localhost:$PORT/health" 2>/dev/null || echo "")
            if echo "$STATUS" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
                echo "      Service healthy after $((i/2))s."
                break
            fi
            printf "."
            sleep 0.5
        done
        echo ""

        # Verify hooks
        echo "[4/5] Verifying hooks..."
        MISSING=""
        for h in recall.sh store.sh register-env.sh register-tools.sh compact.sh; do
            [ ! -f "$HOOKS_DIR/$h" ] && MISSING="$MISSING $h"
        done
        if [ -n "$MISSING" ]; then
            echo "      ERROR: Missing hooks:$MISSING"
            exit 1
        fi
        chmod +x "$HOOKS_DIR/"*.sh
        echo "      All hooks present and executable."

        # Patch settings.json
        echo "[5/5] Patching ~/.claude/settings.json..."
        python3 - "$CLAUDE_SETTINGS" << 'PYEOF'
import json, sys, os
settings_path = sys.argv[1]
# Use AGENTMEM_HOOKS_DIR env override pattern so the path can be changed
# without re-running setup. Falls back to ~/code/agentmem/claude-code/hooks.
def _hook(script, timeout):
    cmd = (
        f'bash -lc \'d="${{AGENTMEM_HOOKS_DIR:-$HOME/code/agentmem/claude-code/hooks}}"; '
        f'f="$d/{script}"; if [ -x "$f" ]; then "$f"; else exit 0; fi\''
    )
    return {"type": "command", "command": cmd, "timeout": timeout}
with open(settings_path) as f: settings = json.load(f)
settings["hooks"] = {
    "SessionStart": [{"matcher": "startup|clear|compact", "hooks": [
        _hook("register-env.sh",   10),
        _hook("register-tools.sh", 10),
    ]}],
    "UserPromptSubmit": [{"hooks": [_hook("recall.sh",  10)]}],
    "PostToolUse":      [{"hooks": [_hook("compact.sh",  5)]}],
    "Stop":             [{"hooks": [_hook("store.sh",   30)]}],
}
with open(settings_path, 'w') as f: json.dump(settings, f, indent=2)
for event, configs in settings["hooks"].items():
    for cfg in configs:
        for h in cfg.get("hooks", []):
            script = h["command"].split('"')[-2].split("/")[-1] if '"' in h["command"] else h["command"]
            print(f"      {event}: {script}")
PYEOF

        echo ""
        echo "=== Setup complete! Open a new Claude Code session to activate. ==="
        ;;

    clean)
        echo "Cleaning orphan uvicorn processes (excluding launchd-managed)..."
        uid="$(id -u)"
        launchd_pid=""
        if launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
            launchd_pid=$(launchctl print "gui/$uid/$SERVICE_LABEL" 2>/dev/null \
                | awk '/^[[:space:]]+pid =/ {print $3}' | head -1)
        fi
        killed=0
        while IFS= read -r pid; do
            [ "$pid" = "$launchd_pid" ] && continue
            ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
            [ "$ppid" = "$launchd_pid" ] && continue
            kill "$pid" 2>/dev/null && killed=$((killed + 1))
        done < <(pgrep -f "uvicorn main:app" 2>/dev/null || true)
        [ "$killed" -gt 0 ] && echo "Killed $killed orphan(s)" || echo "Nothing to clean"
        ;;

    help|--help|-h)
        print_banner
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Service control:"
        echo "  start          Start via launchd (falls back to direct if not installed)"
        echo "  start-fg       Start in foreground — used by launchd itself (Ctrl+C to stop)"
        echo "  stop           Stop via launchd (KeepAlive will restart; use 'disable' to prevent)"
        echo "  restart        Restart via launchd kickstart -k"
        echo "  restart --force  Restart even with active connections"
        echo "  status         Show running status + memory stats"
        echo ""
        echo "Persistent service (auto-start on login):"
        echo "  enable         Install as launchd service"
        echo "  disable        Remove launchd service"
        echo ""
        echo "Monitoring:"
        echo "  health         Health check"
        echo "  logs           View stdout/stderr logs"
        echo ""
        echo "Maintenance:"
        echo "  clean          Kill orphan uvicorn processes"
        echo "  setup          Install/refresh Claude Code hooks"
        echo ""
        echo "Benchmarking:"
        echo "  bench start    Start isolated benchmark instance (:$BENCH_PORT, redis db=$BENCH_DB)"
        echo "  bench fg       Start benchmark in foreground"
        echo "  bench stop     Stop benchmark instance"
        echo "  bench flush    Wipe Redis db=$BENCH_DB (benchmark data only)"
        echo ""
        echo "Endpoints (port $PORT):"
        echo "  API:  http://localhost:$PORT"
        echo "  MCP:  http://localhost:$PORT/mcp"
        echo "  UI:   http://localhost:$PORT/static/index.html"
        echo ""
        ;;

    *)
        echo "Unknown command: ${1}"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
