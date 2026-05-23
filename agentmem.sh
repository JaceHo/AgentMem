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
    local skip_initial="${3:-0}"  # seconds to skip before first poll
    local t0; t0=$(date +%s%3N)
    echo -n "  Waiting for service"
    # Skip initial wait (old process dying / launchd starting new one)
    if [ "$skip_initial" -gt 0 ]; then
        sleep "$skip_initial"
    fi
    for i in $(seq 1 "$max"); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            local t1; t1=$(date +%s%3N)
            printf " ready (%dms)\n" "$((t1 - t0))"
            return 0
        fi
        printf "."
        sleep 0.1
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
    # Pass --nowait if calling after _wait_ready already confirmed health
    local nowait=false; [ "${1:-}" = "--nowait" ] && nowait=true
    if ! $nowait; then
        for i in $(seq 1 40); do
            curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            sleep 0.5
        done
    fi
    uv run python "$SCRIPT_DIR/scripts/register-skills.py" \
        && echo "[agentmem] skills+agents registered" \
        || echo "[agentmem] skills+agents registration failed"
}

_auto_connect() {
    # Auto-connect all detected agents. Pass --nowait if already confirmed healthy.
    local nowait=false; [ "${1:-}" = "--nowait" ] && nowait=true
    if ! $nowait; then
        for i in $(seq 1 40); do
            curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            sleep 0.5
        done
    fi
    echo "[agentmem] auto-connecting detected agents..."
    uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import run_connect
run_connect(all_agents=True)
" 2>&1 | sed 's/^/  /'
}

# ── Commands ──────────────────────────────────────────────────────────────────

case "${1:-help}" in

    start)
        print_banner
        check_deps

        # Parse flags
        NO_AUTO=false
        for arg in "$@"; do [ "$arg" = "--no-auto" ] && NO_AUTO=true; done

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
            if ! $NO_AUTO; then
                _auto_connect &
            else
                echo "  (--no-auto: skipping agent auto-connect)"
            fi
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

        # Parse flags
        NO_AUTO=false
        for arg in "$@"; do [ "$arg" = "--no-auto" ] && NO_AUTO=true; done

        echo "  Starting in foreground on :$PORT"
        echo ""
        _kill_stale
        ( _register_capabilities & )  # double-fork: orphan to launchd, not zombie of uv run
        if ! $NO_AUTO; then
            ( _auto_connect & )
        else
            echo "  (--no-auto: skipping agent auto-connect)"
        fi
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
            pkill -f "uvicorn main:app" || true
            echo "Stopped"
        else
            echo "Not running"
        fi
        ;;

    restart)
        FORCE=false; NO_AUTO=false
        for arg in "$@"; do
            [ "$arg" = "--force" ] && FORCE=true
            [ "$arg" = "--no-auto" ] && NO_AUTO=true
        done

        uid="$(id -u)"
        if ! $FORCE && ! _check_connections; then
            echo ""
            echo "Restart aborted — active connections will be dropped."
            echo "Use --force to restart anyway."
            exit 1
        fi

        t_start=$SECONDS

        # ── 1. Kill existing process directly so port frees immediately ────────
        OLD_PID=$(lsof -ti "tcp:$PORT" 2>/dev/null | head -1 || true)
        if [ -n "$OLD_PID" ]; then
            printf "  Killing pid %s ... " "$OLD_PID"
            kill "$OLD_PID" 2>/dev/null || true
            for _i in 1 2 3 4 5 6 7 8 9 10; do
                lsof -ti "tcp:$PORT" >/dev/null 2>&1 || break
                sleep 0.1
            done
            echo "done"
        fi

        # ── 2. kickstart -k: launchd owns the lifecycle, start fresh ──────────
        if ! launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
            echo "  ERROR: launchd service not loaded — run 'enable' first"
            exit 1
        fi
        mkdir -p "$LOG_DIR" && touch "$LOG_STDOUT" "$LOG_STDERR"
        LOG_POS=$(wc -l < "$LOG_STDOUT" | tr -d ' ')
        ERR_POS=$(wc -l < "$LOG_STDERR" | tr -d ' ')
        echo "  Kicking launchd..."
        launchctl kickstart -k "gui/$uid/$SERVICE_LABEL" || true

        # ── 3. Stream new log lines while polling /health ──────────────────────
        echo "  Waiting for /health:"
        READY=false
        for _i in $(seq 1 100); do
            # show any new stdout lines from the service
            NEW=$(awk "NR>$LOG_POS" "$LOG_STDOUT" 2>/dev/null || true)
            if [ -n "$NEW" ]; then
                printf '%s\n' "$NEW" | sed 's/^/    /'
                LOG_POS=$(wc -l < "$LOG_STDOUT" | tr -d ' ')
            fi
            if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                READY=true
                break
            fi
            sleep 0.2
        done

        # flush remaining lines
        awk "NR>$LOG_POS" "$LOG_STDOUT" 2>/dev/null | sed 's/^/    /' || true

        if [ "$READY" = false ]; then
            echo "  FAILED after $((SECONDS - t_start))s — stderr:"
            awk "NR>$ERR_POS" "$LOG_STDERR" 2>/dev/null | tail -20 | sed 's/^/    /' || true
            exit 1
        fi
        echo "  ✓ Up in $((SECONDS - t_start))s"

        # ── 4+5. Register skills + auto-connect in background ─────────────────
        (
            uv run python "$SCRIPT_DIR/scripts/register-skills.py" \
                >> "$LOG_DIR/setup.log" 2>&1 || true
            if ! $NO_AUTO; then
                uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import run_connect
run_connect(all_agents=True)
" >> "$LOG_DIR/setup.log" 2>&1 || true
            fi
            echo "[$(date '+%H:%M:%S')] post-restart setup done" >> "$LOG_DIR/setup.log"
        ) &
        disown
        echo "  Post-restart setup running in background (tail $LOG_DIR/setup.log)"
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
        HOOKS_DIR="$SCRIPT_DIR/claude-code/hooks"
        CLAUDE_SETTINGS="$HOME/.claude/settings.json"
        MCP_URL="http://localhost:$PORT"
        MCP_SERVER_PY="$SCRIPT_DIR/mcp_server.py"

        # ── Parse --agent flag ────────────────────────────────────────────────
        SETUP_AGENT=""
        for arg in "${@:2}"; do
            case "$arg" in
                --agent=*) SETUP_AGENT="${arg#--agent=}" ;;
                --agent)   : ;;  # next arg handled below
            esac
        done
        # handle "--agent foo" (two separate args)
        prev=""
        for arg in "${@:2}"; do
            [ "$prev" = "--agent" ] && SETUP_AGENT="$arg"
            prev="$arg"
        done

        # ── Helper: write JSON config merging mcpServers key ─────────────────
        _write_mcp_json() {
            local cfg_file="$1"
            local key="$2"       # mcpServers | servers | mcp
            local entry="$3"     # JSON string for the agentmem entry
            local dir
            dir="$(dirname "$cfg_file")"
            mkdir -p "$dir"
            python3 - "$cfg_file" "$key" "$entry" << 'PYEOF'
import json, sys, os
cfg_path, key, entry_json = sys.argv[1], sys.argv[2], sys.argv[3]
entry = json.loads(entry_json)
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        try:
            cfg = json.load(f)
        except Exception:
            cfg = {}
else:
    cfg = {}
if key not in cfg or not isinstance(cfg[key], dict):
    cfg[key] = {}
cfg[key]["agentmem"] = entry
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"      Wrote {cfg_path}")
PYEOF
        }

        # ── Helper: write YAML mcpServers entry (Continue.dev) ───────────────
        _write_continue_yaml() {
            local cfg_file="$1"
            local dir
            dir="$(dirname "$cfg_file")"
            mkdir -p "$dir"
            python3 - "$cfg_file" "$MCP_SERVER_PY" << 'PYEOF'
import sys, os, re
cfg_path = sys.argv[1]
server_py = sys.argv[2]
block = (
    "\n  - name: agentmem\n"
    f"    command: python\n"
    f"    args:\n"
    f"      - \"{server_py}\"\n"
)
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        content = f.read()
    if "name: agentmem" in content:
        print(f"      {cfg_path} already has agentmem entry (skipped)")
        sys.exit(0)
    if "mcpServers:" in content:
        content = content.rstrip() + block
    else:
        content = content.rstrip() + "\nmcpServers:" + block
else:
    content = "mcpServers:" + block
with open(cfg_path, "w") as f:
    f.write(content)
print(f"      Wrote {cfg_path}")
PYEOF
        }

        # ── Helper: write TOML entry (Codex CLI) ─────────────────────────────
        _write_codex_toml() {
            local cfg_file="$1"
            local dir
            dir="$(dirname "$cfg_file")"
            mkdir -p "$dir"
            python3 - "$cfg_file" "$MCP_SERVER_PY" << 'PYEOF'
import sys, os
cfg_path = sys.argv[1]
server_py = sys.argv[2]
block = (
    '\n[mcp_servers.agentmem]\n'
    'command = "python"\n'
    f'args = ["{server_py}"]\n'
)
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        content = f.read()
    if "[mcp_servers.agentmem]" in content:
        print(f"      {cfg_path} already has agentmem entry (skipped)")
        sys.exit(0)
    content = content.rstrip() + block
else:
    content = block.lstrip()
with open(cfg_path, "w") as f:
    f.write(content)
print(f"      Wrote {cfg_path}")
PYEOF
        }

        # ── Per-agent setup functions ─────────────────────────────────────────

        _setup_claude() {
            echo "  [claude-code] Installing hooks + HTTP MCP config..."
            # Hooks (full lifecycle: recall, store, compact, register)
            if [ -f "$CLAUDE_SETTINGS" ]; then
                python3 - "$CLAUDE_SETTINGS" << 'PYEOF'
import json, sys
settings_path = sys.argv[1]
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
with open(settings_path, "w") as f: json.dump(settings, f, indent=2)
print("      Hooks installed in ~/.claude/settings.json")
PYEOF
            else
                echo "      WARNING: ~/.claude/settings.json not found — skipping hooks"
            fi
            # Also write project .mcp.json with HTTP transport
            _write_mcp_json "$HOME/.claude.json" "mcpServers" \
                "{\"type\":\"http\",\"url\":\"$MCP_URL/mcp\"}"
            echo "      → Restart Claude Code to activate."
        }

        _setup_cursor() {
            echo "  [cursor] Writing .cursor/mcp.json (HTTP transport)..."
            _write_mcp_json "$HOME/.cursor/mcp.json" "mcpServers" \
                "{\"url\":\"$MCP_URL/mcp\"}"
            echo "      → Reload Cursor MCP settings (Cmd+Shift+P → MCP: Reload)."
        }

        _setup_windsurf() {
            echo "  [windsurf] Writing ~/.codeium/windsurf/mcp_config.json (SSE transport)..."
            _write_mcp_json "$HOME/.codeium/windsurf/mcp_config.json" "mcpServers" \
                "{\"serverUrl\":\"$MCP_URL/mcp/sse\"}"
            echo "      → Restart Windsurf to pick up the new server."
        }

        _setup_copilot() {
            echo "  [copilot] Writing ~/.config/Code/User/mcp.json (HTTP transport)..."
            _write_mcp_json "$HOME/.config/Code/User/mcp.json" "servers" \
                "{\"type\":\"http\",\"url\":\"$MCP_URL/mcp\"}"
            echo "      → Run 'MCP: Enable Server (agentmem)' in VS Code Command Palette."
        }

        _setup_zed() {
            echo "  [zed] Patching ~/.config/zed/settings.json (context_servers key)..."
            local cfg="$HOME/.config/zed/settings.json"
            mkdir -p "$(dirname "$cfg")"
            python3 - "$cfg" "$MCP_URL" << 'PYEOF'
import json, sys, os
cfg_path = sys.argv[1]
mcp_url = sys.argv[2]
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        try:
            cfg = json.load(f)
        except Exception:
            cfg = {}
else:
    cfg = {}
if "context_servers" not in cfg:
    cfg["context_servers"] = {}
cfg["context_servers"]["agentmem"] = {"url": f"{mcp_url}/mcp/sse"}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"      Wrote context_servers.agentmem to {cfg_path}")
PYEOF
            echo "      → Restart Zed or reload context servers."
        }

        _setup_continue() {
            echo "  [continue] Writing ~/.continue/config.yaml (stdio transport)..."
            _write_continue_yaml "$HOME/.continue/config.yaml"
            echo "      → Reload Continue.dev extension."
        }

        _setup_augment() {
            echo "  [augment] Writing VS Code user settings augment.advanced.mcpServers..."
            local vscode_settings="$HOME/Library/Application Support/Code/User/settings.json"
            if [ ! -f "$vscode_settings" ]; then
                vscode_settings="$HOME/.config/Code/User/settings.json"
            fi
            python3 - "$vscode_settings" "$MCP_URL" << 'PYEOF'
import json, sys, os
cfg_path = sys.argv[1]
mcp_url = sys.argv[2]
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        try:
            cfg = json.load(f)
        except Exception:
            cfg = {}
else:
    cfg = {}
adv = cfg.setdefault("augment.advanced", {})
servers = adv.setdefault("mcpServers", {})
servers["agentmem"] = {"type": "sse", "url": f"{mcp_url}/mcp/sse"}
cfg["augment.advanced"] = adv
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"      Wrote augment.advanced.mcpServers.agentmem to {cfg_path}")
PYEOF
            echo "      → Restart VS Code to pick up Augment MCP config."
        }

        _setup_codex() {
            echo "  [codex] Writing ~/.codex/config.toml (stdio transport)..."
            _write_codex_toml "$HOME/.codex/config.toml"
            echo "      → Restart Codex CLI to load new MCP server."
        }

        _setup_cline() {
            echo "  [cline] Writing ~/.cline/cline_mcp_settings.json (stdio transport)..."
            _write_mcp_json "$HOME/.cline/cline_mcp_settings.json" "mcpServers" \
                "{\"command\":\"python\",\"args\":[\"$MCP_SERVER_PY\"]}"
            echo "      → Reload Cline MCP servers in VS Code settings."
        }

        _setup_kilo() {
            echo "  [kilo-code] Writing ~/.config/kilo/kilo.jsonc (stdio transport)..."
            _write_mcp_json "$HOME/.config/kilo/kilo.jsonc" "mcpServers" \
                "{\"type\":\"stdio\",\"command\":\"python\",\"args\":[\"$MCP_SERVER_PY\"]}"
            echo "      → Reload Kilo Code MCP servers."
        }

        _setup_kiro() {
            echo "  [kiro] Writing ~/.kiro/settings/mcp.json (stdio transport)..."
            _write_mcp_json "$HOME/.kiro/settings/mcp.json" "mcpServers" \
                "{\"command\":\"python\",\"args\":[\"$MCP_SERVER_PY\"]}"
            echo "      → Restart Kiro to load new MCP server."
        }

        _setup_antigravity() {
            echo "  [antigravity] Writing ~/.gemini/antigravity/mcp_config.json (SSE transport)..."
            _write_mcp_json "$HOME/.gemini/antigravity/mcp_config.json" "mcpServers" \
                "{\"serverUrl\":\"$MCP_URL/mcp/sse\"}"
            echo "      → Restart Antigravity to load new MCP server."
        }

        _setup_opencode() {
            echo "  [opencode] Writing ~/.config/opencode/opencode.json (SSE transport)..."
            local cfg="$HOME/.config/opencode/opencode.json"
            mkdir -p "$(dirname "$cfg")"
            python3 - "$cfg" "$MCP_URL" << 'PYEOF'
import json, sys, os
cfg_path = sys.argv[1]
mcp_url = sys.argv[2]
if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
        try:
            cfg = json.load(f)
        except Exception:
            cfg = {}
else:
    cfg = {}
if "mcp" not in cfg or not isinstance(cfg["mcp"], dict):
    cfg["mcp"] = {}
cfg["mcp"]["agentmem"] = {"type": "remote", "url": f"{mcp_url}/mcp/sse", "enabled": True}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"      Wrote mcp.agentmem to {cfg_path}")
PYEOF
            echo "      → Restart opencode to load new MCP server."
        }

        _detect_agents() {
            local detected=()
            [ -d "$HOME/.cursor" ] || [ -f "$HOME/.cursor/mcp.json" ]   && detected+=(cursor)
            [ -d "$HOME/.codeium/windsurf" ]                              && detected+=(windsurf)
            command -v code &>/dev/null                                   && detected+=(copilot)
            [ -d "$HOME/.config/zed" ] || [ -d "/Applications/Zed.app" ] && detected+=(zed)
            [ -d "$HOME/.continue" ]                                      && detected+=(continue)
            [ -d "$HOME/.cline" ]                                         && detected+=(cline)
            command -v codex &>/dev/null                                  && detected+=(codex)
            [ -d "$HOME/.config/kilo" ]                                   && detected+=(kilo)
            [ -d "$HOME/.kiro" ]                                          && detected+=(kiro)
            [ -d "$HOME/.gemini/antigravity" ]                            && detected+=(antigravity)
            command -v opencode &>/dev/null                               && detected+=(opencode)
            echo "${detected[@]}"
        }

        _run_agent_setup() {
            local agent="$1"
            case "$agent" in
                claude|claude-code)  _setup_claude ;;
                cursor)              _setup_cursor ;;
                windsurf)            _setup_windsurf ;;
                copilot|github-copilot) _setup_copilot ;;
                zed)                 _setup_zed ;;
                continue|continue.dev) _setup_continue ;;
                augment)             _setup_augment ;;
                codex)               _setup_codex ;;
                cline)               _setup_cline ;;
                kilo|kilo-code)      _setup_kilo ;;
                kiro)                _setup_kiro ;;
                antigravity)         _setup_antigravity ;;
                opencode)            _setup_opencode ;;
                aider)
                    echo "  [aider] No native MCP support. Use the stdio server from another agent or"
                    echo "          the community mcpm-aider fork: https://github.com/mcpm/aider-mcp-client"
                    ;;
                *)
                    echo "  Unknown agent: $agent"
                    echo "  Valid agents: claude, cursor, windsurf, copilot, zed, continue, augment,"
                    echo "                codex, cline, kilo, kiro, antigravity, opencode, aider"
                    ;;
            esac
        }

        # ── Main setup flow ───────────────────────────────────────────────────
        print_banner
        echo ""

        # Always run Claude Code hooks setup (it's the primary agent)
        echo "[1/4] Checking launchd plist..."
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

        echo "[2/4] Reloading launchd service..."
        launchctl unload "$PLIST_DST" 2>/dev/null || true
        sleep 1
        launchctl load "$PLIST_DST"
        echo "      Service loaded."

        echo "[3/4] Waiting for service health (up to 40s)..."
        sleep 1
        for i in $(seq 1 200); do
            STATUS=$(curl -sf -m 1 "http://localhost:$PORT/health" 2>/dev/null || echo "")
            if echo "$STATUS" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
                echo "      Service healthy (${i}00ms)."
                break
            fi
            printf "."
            sleep 0.1
        done
        echo ""

        echo "[4/4] Configuring agents..."
        if [ -z "$SETUP_AGENT" ]; then
            # Default: Claude Code only (backward compat)
            MISSING=""
            for h in recall.sh store.sh register-env.sh register-tools.sh compact.sh; do
                [ ! -f "$HOOKS_DIR/$h" ] && MISSING="$MISSING $h"
            done
            if [ -n "$MISSING" ]; then
                echo "      ERROR: Missing hooks:$MISSING"
                exit 1
            fi
            chmod +x "$HOOKS_DIR/"*.sh
            _setup_claude
        elif [ "$SETUP_AGENT" = "all" ]; then
            # Claude Code always
            chmod +x "$HOOKS_DIR/"*.sh 2>/dev/null || true
            _setup_claude
            echo ""
            echo "  Auto-detecting installed agents..."
            DETECTED=$(_detect_agents)
            if [ -z "$DETECTED" ]; then
                echo "  No additional agents detected."
            else
                for agent in $DETECTED; do
                    echo ""
                    _run_agent_setup "$agent"
                done
            fi
        else
            _run_agent_setup "$SETUP_AGENT"
        fi

        echo ""
        echo "════════════════════════════════════════"
        echo "  Setup complete!"
        echo "  MCP endpoints:"
        echo "    Streamable HTTP : $MCP_URL/mcp"
        echo "    SSE (legacy)    : $MCP_URL/mcp/sse"
        echo "    Stdio           : python $MCP_SERVER_PY"
        echo "    System prompt   : $MCP_URL/system-prompt"
        echo "════════════════════════════════════════"
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
        echo "  start          Start via launchd (auto-connects all detected agents)"
        echo "  start --no-auto  Start without auto-connecting agents"
        echo "  start-fg       Start in foreground — used by launchd itself (Ctrl+C to stop)"
        echo "  start-fg --no-auto  Foreground without auto-connect"
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
        echo "  clean                Kill orphan uvicorn processes"
        echo "  setup                Install/refresh Claude Code hooks (default)"
        echo "  setup --agent all    Auto-detect + configure all installed agents"
        echo "  setup --agent NAME   Configure a specific agent:"
        echo "                         claude, cursor, windsurf, copilot, zed,"
        echo "                         continue, augment, codex, cline, kilo,"
        echo "                         kiro, antigravity, opencode, hermes, openclaw, trae, trae-cn"
        echo ""
        echo "Agent install (one-shot):"
        echo "  hermes         Install memory plugin for Hermes Agent"
        echo "  openclaw       Install lifecycle plugin for OpenClaw"
        echo "  trae           Configure MCP server for Trae IDE"
        echo "  trae-cn        Configure MCP server for Trae CN IDE"
        echo ""
        echo "Benchmarking:"
        echo "  bench start    Start isolated benchmark instance (:$BENCH_PORT, redis db=$BENCH_DB)"
        echo "  bench fg       Start benchmark in foreground"
        echo "  bench stop     Stop benchmark instance"
        echo "  bench flush    Wipe Redis db=$BENCH_DB (benchmark data only)"
        echo ""
        echo "Endpoints (port $PORT):"
        echo "  API:           http://localhost:$PORT"
        echo "  MCP (HTTP):    http://localhost:$PORT/mcp"
        echo "  MCP (SSE):     http://localhost:$PORT/mcp/sse"
        echo "  System prompt: http://localhost:$PORT/system-prompt"
        echo "  UI:            http://localhost:$PORT/static/index.html"
        echo ""
        ;;

    hermes)
        print_banner
        echo "Installing agentmem for Hermes Agent..."
        echo ""
        uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import detect_hermes, connect_hermes
if not detect_hermes():
    print('Hermes Agent not detected (no ~/.hermes directory).')
    print('Install with: pip install hermes-agent && hermes postinstall')
    sys.exit(1)
result = connect_hermes(force='--force' in sys.argv)
print(result)
" "$@"
        ;;

    openclaw)
        print_banner
        echo "Installing agentmem for OpenClaw..."
        echo ""
        uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import detect_openclaw, connect_openclaw
if not detect_openclaw():
    print('OpenClaw not detected (no ~/.openclaw directory).')
    print('Install with: npm install -g openclaw')
    sys.exit(1)
result = connect_openclaw(force='--force' in sys.argv)
print(result)
" "$@"
        ;;

    trae)
        print_banner
        echo "Configuring agentmem MCP for Trae IDE..."
        echo ""
        uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import detect_trae, connect_trae
if not detect_trae():
    print('Trae IDE not detected (no ~/.trae directory).')
    print('Install from: https://trae.ai')
    sys.exit(1)
result = connect_trae(force='--force' in sys.argv)
print(result)
" "$@"
        ;;

    trae-cn)
        print_banner
        echo "Configuring agentmem MCP for Trae CN IDE..."
        echo ""
        uv run python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from connect import detect_trae_cn, connect_trae_cn
if not detect_trae_cn():
    print('Trae CN IDE not detected (no ~/.trae-cn directory).')
    print('Install from: https://trae.ai')
    sys.exit(1)
result = connect_trae_cn(force='--force' in sys.argv)
print(result)
" "$@"
        ;;

    *)
        echo "Unknown command: ${1}"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
