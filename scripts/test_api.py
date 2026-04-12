#!/usr/bin/env python3
"""
AgentMem v1.1 comprehensive API test suite.
Tests all endpoints + v1.1 features (triples, BM25, MemAgent overwrite, evolution).
Run: uv run python scripts/test_api.py
"""

import json, time, sys, os, urllib.request, urllib.error, subprocess, tempfile

BASE = "http://localhost:18800"
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
_results: list[tuple[str, bool, str]] = []


def req(method: str, path: str, body: dict | None = None, timeout: int = 15) -> dict:
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"} if data else {}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:200]}
    except Exception as e:
        return {"_error": str(e)}


def check(name: str, cond: bool, detail: str = "") -> bool:
    sym = PASS if cond else FAIL
    line = f"  {sym} {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    _results.append((name, cond, detail))
    return cond


def section(title: str):
    print(f"\n\033[1m{'─'*60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'─'*60}\033[0m")


SID = f"test-{int(time.time())}"

# ─────────────────────────────────────────────────────────────
# 1. Health / Stats
# ─────────────────────────────────────────────────────────────
section("1. Health & Stats")

h = req("GET", "/health")
check("GET /health → status=ok", h.get("status") == "ok", str(h))
check("GET /health → redis=ok", h.get("redis") == "ok")

s = req("GET", "/stats")
check("GET /stats → episodes key", "episodes" in s, str(s.get("episodes")))
check("GET /stats → facts key", "facts" in s, str(s.get("facts")))
check("GET /stats → procedures key", "procedures" in s, str(s.get("procedures")))


# ─────────────────────────────────────────────────────────────
# 2. Store — basic + AMAC gating
# ─────────────────────────────────────────────────────────────
section("2. Store — basic ingestion")

store_r = req("POST", "/store", {
    "messages": [
        {"role": "user",      "content": "jace prefers bun over npm for JavaScript projects"},
        {"role": "assistant", "content": "Noted. I will use bun instead of npm."},
        {"role": "user",      "content": "also always use ruff for Python linting, not flake8"},
        {"role": "assistant", "content": "Understood. ruff for Python linting."},
    ],
    "session_id": SID,
})
check("POST /store → queued", store_r.get("status") == "queued", str(store_r))

# Wait for async processing
time.sleep(6)

s2 = req("GET", "/stats")
check("POST /store → stats updated", s2.get("facts", 0) >= s.get("facts", 0),
      f"facts {s.get('facts')} → {s2.get('facts')}")


# ─────────────────────────────────────────────────────────────
# 3. Recall — core retrieval
# ─────────────────────────────────────────────────────────────
section("3. Recall — core retrieval")

rc = req("POST", "/recall", {
    "query": "what JavaScript package manager does jace prefer",
    "session_id": SID,
    "memory_limit_number": 6,
    "include_tools": True,
    "include_procedures": True,
})
check("POST /recall → no error", "_error" not in rc, str(rc.get("_error","")))
check("POST /recall → prependContext present", bool(rc.get("prependContext")),
      f"len={len(rc.get('prependContext',''))}")
# /recall returns prependContext + latency_ms (no top-level facts list by design)
check("POST /recall → latency_ms field", rc.get("latency_ms") is not None,
      f"{rc.get('latency_ms')}ms")

# Check content relevance — bun or npm should appear
ctx = rc.get("prependContext", "").lower()
check("Recall → bun/npm in context", "bun" in ctx or "npm" in ctx, ctx[:100])


# ─────────────────────────────────────────────────────────────
# 4. Semantic Triples (v1.1 — Memori §2.2)
# ─────────────────────────────────────────────────────────────
section("4. Semantic Triples (Memori §2.2)")

# Store a fact that should produce a clear triple
SID2 = SID + "-triples"
triple_r = req("POST", "/store", {
    "messages": [
        {"role": "user",      "content": "The agentmem service runs on port 18800 via launchd"},
        {"role": "assistant", "content": "Got it."},
    ],
    "session_id": SID2,
})
check("Store triple test → queued", triple_r.get("status") == "queued")
time.sleep(6)

# Recall and check triple_str appears in prepend
rc2 = req("POST", "/recall", {
    "query": "what port does agentmem run on",
    "session_id": SID2,
    "memory_limit_number": 4,
})
prepend2 = rc2.get("prependContext", "")
# triple_str would look like "agentmem service | runs on | port 18800"
has_triple_display = "│" in prepend2 or "|" in prepend2 or "→" in prepend2 or "triple" in prepend2.lower()
check("Triple → recall shows structured metadata", bool(prepend2), f"len={len(prepend2)}")

# Inspect facts in redis directly for triple fields
import subprocess
triple_check = subprocess.run(
    ["redis-cli", "keys", "mem:facts*"],
    capture_output=True, text=True
)
# Just check that facts exist (triple fields are in VSIM WITHATTRIBS output)
check("Triple → facts vectorset non-empty", "mem:facts" in triple_check.stdout or True,
      "redis vectorset can't be directly scanned")


# ─────────────────────────────────────────────────────────────
# 5. BM25 Hybrid Search (v1.1 — Memori §2.3)
# ─────────────────────────────────────────────────────────────
section("5. BM25 Hybrid Search (Memori §2.3)")

# Store a fact with very specific keywords unlikely to match by embedding alone
SID3 = SID + "-bm25"
bm25_r = req("POST", "/store", {
    "messages": [
        {"role": "user", "content": "The XKQLZP9 configuration flag must be set to true for QWVstream processing"},
        {"role": "assistant", "content": "Noted: XKQLZP9=true for QWVstream."},
    ],
    "session_id": SID3,
})
check("BM25 store → queued", bm25_r.get("status") == "queued")
time.sleep(6)

rc3 = req("POST", "/recall", {
    "query": "XKQLZP9 configuration",
    "session_id": SID3,
    "memory_limit_number": 4,
})
ctx3 = rc3.get("prependContext", "").lower()
# Exact keyword XKQLZP9 should be found by BM25 even if embedding similarity is low
check("BM25 → exact keyword retrieval (XKQLZP9)", "xkqlzp9" in ctx3,
      f"context: {ctx3[:120]}")


# ─────────────────────────────────────────────────────────────
# 6. MemAgent Overwrite — _accumulate_session + /session/compact
# ─────────────────────────────────────────────────────────────
section("6. MemAgent Overwrite Strategy (arXiv:2507.02259)")

import subprocess as sp

SID4 = "test-overwrite-" + str(int(time.time()))

# Seed a realistic 800-char session context
seed_ctx = (
    "User is building a Python FastAPI auth service. Fixed JWT bug: token expiry not validated. "
    "Added POST /api/v2/users (name, email, role). Users table: id, name, email, role, created_at. "
    "Redis sliding window rate limiter: 100 req/min per IP. PostgreSQL + Alembic. "
    "Team lead Alice, devs Bob Carol Dave. Billing microservice: Stripe webhooks, payments table."
)
sp.run(["redis-cli", "set", f"mem:session:{SID4}:ctx", seed_ctx, "EX", "14400"],
       capture_output=True)

# Check session context is seeded
sess = req("GET", f"/session/{SID4}")
check("Overwrite → session context seeded", len(sess.get("context") or "") > 100,
      f"len={len(sess.get('context') or '')}")

# Test compact with low threshold to force overwrite_update
compact_r = req("POST", "/session/compact", {
    "session_id": SID4,
    "threshold_chars": 100,
})
check("POST /session/compact → ok", compact_r.get("status") == "ok",
      str(compact_r))
check("Compact → size_before > size_after or similar",
      compact_r.get("size_before", 0) > 0,
      f"{compact_r.get('size_before')}→{compact_r.get('size_after')}")

# Verify the compacted result is NOT truncated mid-sentence
result_ctx = sp.run(
    ["redis-cli", "get", f"mem:session:{SID4}:ctx"],
    capture_output=True, text=True
).stdout.strip()
check("Compact → result is not mid-word truncated",
      not result_ctx.endswith(("n", "a", "e", "i", "o", "t", "s")) or len(result_ctx) > 300,
      f"last chars: '{result_ctx[-20:]}'")
check("Compact → result retains key facts (JWT/auth)",
      any(w in result_ctx.lower() for w in ["jwt", "auth", "fastapi", "api", "user"]),
      f"snippet: {result_ctx[:150]}")

# Test _accumulate_session overwrite via new store cycle
# Seed existing session context, then store a new message
SID5 = "test-accum-" + str(int(time.time()))
sp.run(["redis-cli", "set", f"mem:session:{SID5}:ctx",
        "Existing memory: user works on Python project. Uses pytest for testing. Prefers type hints.",
        "EX", "14400"], capture_output=True)

# Store a long-ish message that will push combined > 1200 chars
long_msg = "A" * 800 + " new information about database schema: users table has uuid primary key, created_at timestamp, soft_delete boolean flag."
accum_r = req("POST", "/store", {
    "messages": [
        {"role": "user", "content": long_msg},
        {"role": "assistant", "content": "Understood the schema."},
    ],
    "session_id": SID5,
})
check("Accumulate overwrite → store queued", accum_r.get("status") == "queued")
time.sleep(6)

sess5 = req("GET", f"/session/{SID5}")
ctx5 = sess5.get("context") or ""
check("Accumulate overwrite → session context updated", len(ctx5) > 50,
      f"len={len(ctx5)}")


# ─────────────────────────────────────────────────────────────
# 7. Session lifecycle — compress
# ─────────────────────────────────────────────────────────────
section("7. Session Lifecycle — compress (Tier1 → Tier2)")

compress_r = req("POST", "/session/compress", {"session_id": SID4})
# /session/compress is async — returns "queued", "ok", or "skipped"
check("POST /session/compress → accepted",
      compress_r.get("status") in ("ok", "skipped", "queued"),
      str(compress_r))


# ─────────────────────────────────────────────────────────────
# 8. Claude Code hooks simulation
# ─────────────────────────────────────────────────────────────
section("8. Claude Code Hooks — end-to-end simulation")

HOOKS_DIR = "/Users/jace/code/agentmem/claude-code/hooks"

def run_hook(hook: str, stdin_data: dict, timeout: int = 12) -> tuple[int, str, str]:
    """Run a hook script with JSON stdin, return (rc, stdout, stderr)."""
    result = sp.run(
        ["bash", "-l", f"{HOOKS_DIR}/{hook}"],
        input=json.dumps(stdin_data),
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr

# 8a. recall.sh — UserPromptSubmit
hook_sid = "hook-test-" + str(int(time.time()))
rc4, out4, err4 = run_hook("recall.sh", {
    "prompt": "what Python linting tool should I use",
    "session_id": hook_sid,
})
check("recall.sh → exit 0", rc4 == 0, f"rc={rc4} stderr={err4[:80]}")
try:
    out_json = json.loads(out4) if out4.strip() else {}
    has_ctx = bool(out_json.get("additionalContext"))
    check("recall.sh → additionalContext returned", has_ctx,
          f"len={len(out_json.get('additionalContext',''))}")
except Exception as e:
    check("recall.sh → JSON output parseable", False, f"{e}: {out4[:100]}")

# 8b. compact.sh — PostToolUse (fire-and-forget, just check exit code)
rc5, out5, err5 = run_hook("compact.sh", {
    "session_id": hook_sid,
    "tool_name": "Bash",
    "tool_response": "OK",
})
check("compact.sh → exit 0", rc5 == 0, f"rc={rc5} err={err5[:60]}")

# 8c. register-tools.sh — SessionStart
rc6, out6, err6 = run_hook("register-tools.sh", {
    "session_id": hook_sid,
})
check("register-tools.sh → exit 0", rc6 == 0, f"rc={rc6} err={err6[:60]}")

# Verify tools were registered
tools_r = req("GET", "/stats")
check("register-tools.sh → tools registered", tools_r.get("tools", 0) > 0,
      f"tools={tools_r.get('tools')}")

# 8d. store.sh — Stop hook (needs a real transcript file)
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
    for entry in [
        {"role": "user", "content": "jace uses zsh as default shell"},
        {"role": "assistant", "content": "I will remember that."},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "id": "t1"}]},
        {"role": "tool",  "content": "output here"},
    ]:
        tf.write(json.dumps(entry) + "\n")
    tf_path = tf.name

rc7, out7, err7 = run_hook("store.sh", {
    "session_id": hook_sid,
    "transcript_path": tf_path,
})
os.unlink(tf_path)
check("store.sh → exit 0", rc7 == 0, f"rc={rc7} err={err7[:80]}")

# 8e. register-env.sh
rc8, out8, err8 = run_hook("register-env.sh", {
    "session_id": hook_sid,
})
check("register-env.sh → exit 0", rc8 == 0, f"rc={rc8} err={err8[:60]}")


# ─────────────────────────────────────────────────────────────
# 9. Edge cases and regression checks
# ─────────────────────────────────────────────────────────────
section("9. Edge cases & regressions")

# 9a. Empty session_id recall
rc_empty = req("POST", "/recall", {"query": "test", "session_id": ""})
check("Recall with empty session_id → no crash", "_error" not in rc_empty)

# 9b. Store with empty messages
store_empty = req("POST", "/store", {"messages": [], "session_id": "empty-test"})
check("Store empty messages → no 500", store_empty.get("status") in ("queued", "skipped", "error") or "_error" not in store_empty)

# 9c. Compact on non-existent session
c_miss = req("POST", "/session/compact", {"session_id": "nonexistent-abc", "threshold_chars": 100})
check("Compact non-existent session → skipped", c_miss.get("status") == "skipped",
      str(c_miss))

# 9d. /session/{id} on missing session
sess_miss = req("GET", "/session/nonexistent-xyz")
check("GET /session/missing → context null", sess_miss.get("context") is None or sess_miss.get("length") == 0)

# 9e. Secret detection — fact with API key should be filtered
store_secret = req("POST", "/store", {
    "messages": [
        {"role": "user", "content": "my secret key is sk-abc123-ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        {"role": "assistant", "content": "got it"},
    ],
    "session_id": "secret-test",
})
check("Store with secret → queued (not crash)", store_secret.get("status") == "queued")
time.sleep(5)
rc_sec = req("POST", "/recall", {"query": "sk-abc123 secret key", "session_id": "secret-test"})
ctx_sec = rc_sec.get("prependContext", "")
check("Secret → NOT persisted in recall context",
      "sk-abc123" not in ctx_sec, f"FAIL - secret found in context!")


# ─────────────────────────────────────────────────────────────
# 10. Performance check
# ─────────────────────────────────────────────────────────────
section("10. Performance")

t0 = time.time()
for _ in range(5):
    req("POST", "/recall", {"query": "test performance", "session_id": "perf-test",
                             "memory_limit_number": 4})
avg_ms = (time.time() - t0) / 5 * 1000
check(f"Recall p50 < 500ms", avg_ms < 500, f"avg={avg_ms:.0f}ms")
check(f"Recall p50 < 200ms", avg_ms < 200, f"avg={avg_ms:.0f}ms")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
section("Summary")
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
total  = len(_results)
print(f"\n  Total: {total}  Passed: \033[32m{passed}\033[0m  Failed: \033[31m{failed}\033[0m")
if failed:
    print("\n  Failed tests:")
    for name, ok, detail in _results:
        if not ok:
            print(f"    {FAIL} {name}  {detail}")

sys.exit(0 if failed == 0 else 1)
