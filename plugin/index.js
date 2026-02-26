/**
 * memos-local-openclaw-plugin
 * Local long-memory via Redis HNSW + MiniLM FastAPI sidecar.
 * Drop-in replacement for memos-cloud-openclaw-plugin.
 */

const DEFAULT_BASE_URL = "http://127.0.0.1:18790";
const DEFAULT_LIMIT    = 6;
const RECALL_TIMEOUT   = 3000;   // ms — block agent start if exceeded
const STORE_TIMEOUT    = 500;    // ms — fire-and-forget, don't block

export default {
  id:          "memos-local-openclaw-plugin",
  name:        "Local Memory (Redis HNSW + MiniLM)",
  description: "Long-term memory via local FastAPI sidecar + Redis vector search. No cloud dependency.",
  kind:        "lifecycle",

  register(api) {
    const cfg    = api.pluginConfig ?? {};
    const log    = api.logger ?? console;
    const base   = (cfg.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    const limit  = cfg.memoryLimitNumber ?? DEFAULT_LIMIT;
    const tag    = "[memos-local]";

    // ── before_agent_start: recall relevant memories ────────────────────────
    api.on("before_agent_start", async (event, ctx) => {
      const query = event?.prompt;
      if (!query) return;

      try {
        const res = await fetch(`${base}/recall`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            query,
            session_id:          ctx?.sessionId ?? ctx?.sessionKey ?? "",
            memory_limit_number: limit,
          }),
          signal: AbortSignal.timeout(RECALL_TIMEOUT),
        });

        if (!res.ok) {
          log.warn?.(`${tag} recall HTTP ${res.status}`);
          return;
        }

        const { prependContext, latency_ms } = await res.json();
        if (prependContext) {
          log.info?.(`${tag} recalled in ${latency_ms}ms`);
          return { prependContext };
        }
      } catch (e) {
        if (e?.name === "TimeoutError") {
          log.warn?.(`${tag} recall timed out (${RECALL_TIMEOUT}ms) — skipping`);
        } else {
          log.warn?.(`${tag} recall error: ${e?.message}`);
        }
      }
    });

    // ── agent_end: store conversation (fire-and-forget) ─────────────────────
    api.on("agent_end", async (event, ctx) => {
      if (!event?.success) return;
      const messages = event?.messages;
      if (!messages?.length) return;

      // Fire and forget — don't await, don't block response
      fetch(`${base}/store`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          messages,
          session_id: ctx?.sessionId ?? ctx?.sessionKey ?? "",
        }),
        signal: AbortSignal.timeout(STORE_TIMEOUT),
      }).catch(() => {}); // intentional silent ignore
    });

    log.info?.(`${tag} registered (base=${base}, limit=${limit})`);
  },
};
