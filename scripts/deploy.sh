#!/usr/bin/env bash
# agentmem deploy channel (outer loop) — closes governance gap t_53791ec9
# "P1 agentmem: active within 30d but no deploy channel (outer loop not laid)"
#
# Path to production on this host:  git commit → scripts/deploy.sh → launchd restart
# → /health gate → deploy record.  No more "prod runs whatever is in the working tree".
#
# Usage:
#   scripts/deploy.sh                 # preflight → gate → stamp → restart → health → record
#   scripts/deploy.sh --dry-run       # every check, no restart (safe on live service)
#   scripts/deploy.sh --rollback      # redeploy last recorded good commit
#   scripts/deploy.sh --health        # just poll /health
#
# Env overrides:
#   ALLOW_DIRTY=1   deploy despite uncommitted changes (logged, discouraged)
#   RUN_TESTS=1     run pytest suite in the gate step (slower, for risky changes)
#   AUTO_ROLLBACK=1 auto-checkout previous good commit if health gate fails
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="ai.agent.memory"
PORT="${AGENTMEM_PORT:-18800}"
HEALTH_URL="http://localhost:${PORT}/health"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"   # model load can take a while after cold start
STATE_DIR="$HOME/.agentmem"
STATE_FILE="$STATE_DIR/deploy_state.json"
LOG_FILE="$STATE_DIR/logs/deploy.log"

log() { printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$LOG_FILE"; }
die() { log "FAIL: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"; }

need git; need curl; need launchctl
mkdir -p "$STATE_DIR/logs"

MODE="deploy"
for arg in "$@"; do
  case "$arg" in
    --dry-run)  MODE="dry-run" ;;
    --rollback) MODE="rollback" ;;
    --health)   MODE="health" ;;
    *) die "unknown flag: $arg (expected --dry-run | --rollback | --health)" ;;
  esac
done

py_bin() {
  if [ -x "$REPO_ROOT/venv/bin/python" ]; then echo "$REPO_ROOT/venv/bin/python"
  elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then echo "$REPO_ROOT/.venv/bin/python"
  else command -v python3; fi
}

health_ok() {
  curl -sf -m 5 "$HEALTH_URL" | "$(py_bin)" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("status")=="ok" else 1)' 2>/dev/null
}

wait_healthy() {  # $1 = what we're waiting for (label used in messages)
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if health_ok; then log "health OK ($1) after $(( $(date +%s) + HEALTH_TIMEOUT - deadline ))s: $(curl -sf -m 5 "$HEALTH_URL" 2>/dev/null | head -c 200)"; return 0; fi
    sleep 3
  done
  die "health gate FAILED for $1: no ok from $HEALTH_URL within ${HEALTH_TIMEOUT}s"
}

read_state_field() {  # $1 = field name
  [ -f "$STATE_FILE" ] && "$(py_bin)" -c "import json,sys; print(json.load(open('$STATE_FILE')).get('$1',''))" 2>/dev/null || true
}

write_state() {  # $1=json fragment written by caller via python
  "$(py_bin)" - "$@" <<'PY'
import json, sys, datetime
path = sys.argv[1]
try: state = json.load(open(path))
except Exception: state = {}
fields = json.loads(sys.argv[2])
state.update(fields)
state["history"] = ([{"ts": state.get("ts"), "commit": state.get("commit"), "status": state.get("status")}]
                    if state.get("commit") else []) + state.get("history", [])
state["history"] = state["history"][:10]
json.dump(state, open(path, "w"), indent=2)
PY
}

kick_and_gate() {  # $1 = commit being deployed
  log "restarting launchd service $LABEL (deploy $1)"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
  wait_healthy "$1"
}

cd "$REPO_ROOT"

case "$MODE" in
  health)
    health_ok && { echo "healthy: $(curl -sf -m 5 "$HEALTH_URL")"; exit 0; } || die "unhealthy"
    ;;

  rollback)
    PREV="$(read_state_field commit)"
    [ -n "$PREV" ] || die "no previous good commit recorded in $STATE_FILE"
    log "ROLLBACK to last good commit $PREV"
    [ "$(git rev-parse HEAD)" = "$PREV" ] || {
      git diff --quiet || die "working tree dirty — commit or stash before rollback"
      git checkout --detach "$PREV"
    }
    kick_and_gate "rollback:$PREV"
    write_state "$STATE_FILE" "{\"ts\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\", \"commit\": \"$PREV\", \"status\": \"rolled-back\"}"
    log "ROLLBACK complete -> $PREV"
    ;;

  dry-run | deploy)
    DRY=0; [ "$MODE" = "dry-run" ] && DRY=1
    HEAD="$(git rev-parse HEAD)"
    SHORT="$(git rev-parse --short HEAD)"

    # 1. preflight ---------------------------------------------------------
    if ! git diff --quiet || ! git diff --cached --quiet; then
      if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
        log "WARN: deploying DIRTY tree (ALLOW_DIRTY=1) at $SHORT"
      else
        die "working tree dirty — commit first (or ALLOW_DIRTY=1 to override)"
      fi
    fi

    # 2. gate: compile + optional tests ------------------------------------
    log "gate: compile check at $SHORT"
    "$(py_bin)" -m compileall -q main.py mcp_server.py api core adapters cli config || die "compile gate failed"
    if [ "${RUN_TESTS:-0}" = "1" ]; then
      log "gate: pytest (RUN_TESTS=1)"
      (cd "$REPO_ROOT" && "$(py_bin)" -m pytest -q tests/ || die "test gate failed")
    fi

    if [ "$DRY" = 1 ]; then
      log "DRY-RUN OK at $SHORT — preflight+gate passed; restart/health/record skipped"
      exit 0
    fi

    # 3. stamp, restart, health-gate, record -------------------------------
    PREV="$(read_state_field commit)"
    write_state "$STATE_FILE" "{\"ts\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\", \"commit\": \"$HEAD\", \"status\": \"deploying\"}"
    if kick_and_gate "$SHORT"; then
      write_state "$STATE_FILE" "{\"status\": \"healthy\"}"
      log "DEPLOY OK: $SHORT (prev: ${PREV:-none})"
    else
      write_state "$STATE_FILE" "{\"status\": \"failed\"}"
      if [ "${AUTO_ROLLBACK:-0}" = "1" ] && [ -n "$PREV" ] && [ "$PREV" != "$HEAD" ]; then
        log "AUTO_ROLLBACK to $PREV"
        git checkout --detach "$PREV" && kick_and_gate "auto-rollback:$PREV" \
          && write_state "$STATE_FILE" "{\"ts\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\", \"commit\": \"$PREV\", \"status\": \"rolled-back\"}" \
          || die "auto-rollback also failed — manual intervention needed"
      else
        die "deploy of $SHORT failed — run: scripts/deploy.sh --rollback (or fix forward)"
      fi
    fi
    ;;
esac