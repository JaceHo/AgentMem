#!/bin/bash
# Memory service startup wrapper.
# Waits for port 18800 to be free before starting uvicorn.
# Fixes "address already in use" on launchd restarts.

PORT=18800
WORKDIR="$(dirname "$0")"
cd "$WORKDIR"

# Kill any stale process still holding the port
OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$OLD_PID" ]; then
  echo "[start.sh] Killing stale PID $OLD_PID on port $PORT"
  kill -9 $OLD_PID 2>/dev/null
  sleep 1
fi

# Wait until port is free (max 8s)
for i in $(seq 1 8); do
  if ! lsof -ti tcp:$PORT >/dev/null 2>&1; then
    break
  fi
  echo "[start.sh] Waiting for port $PORT to be free (attempt $i)…"
  sleep 1
done

exec venv/bin/python -m uvicorn main:app \
  --host 127.0.0.1 \
  --port $PORT \
  --log-level info \
  --timeout-graceful-shutdown 0
