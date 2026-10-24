# Deploy Runbook (outer loop)

Closes governance finding **t_53791ec9** — *"P1 agentmem: active within 30d but no
deploy channel (outer loop not laid)"*. Before this, the launchd service
`ai.agent.memory` ran whatever happened to be in the working tree: no gate, no
version record, no health check, no rollback.

## The channel

```
git commit ──▶ scripts/deploy.sh ──▶ launchctl kickstart ai.agent.memory
                    │                        │
                    ├─ preflight (clean tree) ├─ /health gate (status == "ok")
                    ├─ compile gate           └─ record → ~/.agentmem/deploy_state.json
                    └─ optional pytest (RUN_TESTS=1)             + logs/deploy.log
```

Runtime state lives **outside the repo** (`~/.agentmem/`), so deploying never
dirties the tree.

## Commands

| Command | What it does |
|---|---|
| `scripts/deploy.sh --health` | Poll `http://localhost:18800/health` once. |
| `scripts/deploy.sh --dry-run` | Preflight + gates only. Never touches the service. Safe anytime. |
| `scripts/deploy.sh` | Full deploy: gate → restart → health-gate → record. |
| `scripts/deploy.sh --rollback` | Checkout last recorded good commit, restart, health-gate, record. |

Env knobs (all optional):

- `ALLOW_DIRTY=1` — deploy with uncommitted changes. Logged with a WARN; only
  for hotfixes.
- `RUN_TESTS=1` — run `pytest tests/` as part of the gate (slower; use for
  risky changes).
- `AUTO_ROLLBACK=1` — if the health gate fails, auto-checkout the previous good
  commit and restart. Off by default: fix-forward is usually better on a dev box.
- `HEALTH_TIMEOUT=90` — seconds to wait for `/health` after restart (model load
  on cold start is the slow part).

## Shipping a change

```bash
# on a feature branch, work done:
git commit -m "fix: ..."
scripts/deploy.sh --dry-run          # gates pass?
scripts/deploy.sh                    # live deploy + health gate
curl -s localhost:18800/health       # confirm version bumped
```

If health never turns ok: `scripts/deploy.sh --rollback`.

## Audit trail

- `~/.agentmem/deploy_state.json` — last deploy (commit, ts, status) + last 10 history entries.
- `~/.agentmem/logs/deploy.log` — one line per event, append-only.

## Notes / follow-ups (not done in t_53791ec9)

- No CI: GitHub Actions workflow deliberately omitted — deploys are local to
  this host, and the branch must not be pushed. Add CI when the repo gets a
  hosted environment.
- Version drift: running service reports `1.1.0` while `pyproject.toml` says
  `0.9.7` (working-tree code ahead of the last release). First real deploy
  through this channel should bump `pyproject.toml` so `/health` version and
  git history line up.
- The plist (`ai.agent.memory.plist`) references `agentmem.sh start-fg`; this
  channel restarts via `launchctl kickstart -k` and does not change how the
  service starts.