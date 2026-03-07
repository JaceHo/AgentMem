#!/usr/bin/env bash
# bench-start.sh — Start isolated AgentMem instance for F1 benchmarking.
#
# Isolation:
#   Port    : 18899   (production: 18800)
#   Redis DB: 15      (production: 0)
#   Log     : /tmp/agentmem-bench.log
#
# Usage:
#   bash bench-start.sh          # start in background
#   bash bench-start.sh --fg     # start in foreground (verbose)
#   bash bench-start.sh --stop   # kill the benchmark service
#   bash bench-start.sh --flush  # wipe Redis db=15 (benchmark data only)

BENCH_PORT=18899
BENCH_DB=15
BENCH_LOG=/tmp/agentmem-bench.log
BENCH_PID=/tmp/agentmem-bench.pid

cd "$(dirname "$0")"

# ── stop ──────────────────────────────────────────────────────────────────────
if [ "$1" = "--stop" ]; then
    if [ -f "$BENCH_PID" ]; then
        PID=$(cat "$BENCH_PID")
        kill "$PID" 2>/dev/null && echo "Stopped benchmark service (pid $PID)" || echo "Already stopped"
        rm -f "$BENCH_PID"
    else
        # Fallback: kill by port
        lsof -ti tcp:$BENCH_PORT | xargs kill -9 2>/dev/null && echo "Killed process on port $BENCH_PORT" || echo "Nothing running on port $BENCH_PORT"
    fi
    exit 0
fi

# ── flush ─────────────────────────────────────────────────────────────────────
if [ "$1" = "--flush" ]; then
    redis-cli -n $BENCH_DB FLUSHDB && echo "Flushed Redis db=$BENCH_DB (benchmark data only)"
    exit 0
fi

# ── kill stale process on bench port ──────────────────────────────────────────
STALE=$(lsof -ti tcp:$BENCH_PORT 2>/dev/null)
if [ -n "$STALE" ]; then
    echo "Killing stale process on port $BENCH_PORT (pid $STALE)..."
    kill -9 $STALE 2>/dev/null
    sleep 1
fi

echo "Starting AgentMem benchmark service..."
echo "  Port    : $BENCH_PORT"
echo "  Redis DB: $BENCH_DB  (redis://localhost:6379/$BENCH_DB)"
echo "  Log     : $BENCH_LOG"
echo ""

export REDIS_URL="redis://localhost:6379/$BENCH_DB"
# Benchmark mode: lower A-MAC gate threshold so all test conversations are stored.
# Production default is 0.30; bench uses 0.05 to admit all factual conversation turns.
export AMAC_THRESHOLD="0.05"

# Use the project venv python (has all deps: redis, fastapi, uvicorn, torch…)
VENV_PYTHON="$(dirname "$0")/venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# ── foreground mode ───────────────────────────────────────────────────────────
if [ "$1" = "--fg" ]; then
    "$VENV_PYTHON" -m uvicorn main:app \
        --host 127.0.0.1 \
        --port $BENCH_PORT \
        --log-level info
    exit 0
fi

# ── background mode ───────────────────────────────────────────────────────────
"$VENV_PYTHON" -m uvicorn main:app \
    --host 127.0.0.1 \
    --port $BENCH_PORT \
    --log-level warning \
    > "$BENCH_LOG" 2>&1 &

echo $! > "$BENCH_PID"
echo "Benchmark service started (pid $!, log: $BENCH_LOG)"

# Wait for readiness
echo -n "Waiting for service..."
for i in $(seq 1 20); do
    sleep 1
    STATUS=$(curl -s http://localhost:$BENCH_PORT/health 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
    if [ "$STATUS" = "ok" ]; then
        echo " ready!"
        echo ""
        echo "Benchmark service is up at http://localhost:$BENCH_PORT"
        echo "Stop with:  bash bench-start.sh --stop"
        echo "Flush DB:   bash bench-start.sh --flush"
        exit 0
    fi
    echo -n "."
done

echo " timeout! Check $BENCH_LOG for errors."
exit 1
