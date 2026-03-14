/**
 * agentmem-openclaw-plugin  v0.5.0
 * AgentMem — local persistent memory via Redis HNSW + MiniLM FastAPI sidecar.
 * Drop-in replacement for memos-cloud-openclaw-plugin.
 *
 * v0.3.0 additions (2026 layered controlled architecture):
 *   - Session Tier 1→2 promotion: calls /session/compress on agent_end
 *     so accumulated session context is crystallised into long-term memory
 *     before the 4h TTL expires and data is lost
 *   - Store still queued in parallel (fire-and-forget)
 *
 * v0.2.0 additions (capability registry):
 *   - Registers tool/skill index on before_agent_start
 *   - Registers environment state on before_agent_start
 *   - Uses /recall-tools for capability-style queries
 */

const DEFAULT_BASE_URL = "http://127.0.0.1:18800";
const DEFAULT_LIMIT    = 6;
const RECALL_TIMEOUT   = 7000;   // ms — block agent start if exceeded (raised from 3000: avg recall is 4-6s)
const STORE_TIMEOUT    = 500;    // ms — fire-and-forget, don't block
const REG_TIMEOUT      = 2000;   // ms — tool/env registration (non-blocking)

export default {
  id:          "memos-local-openclaw-plugin",
  name:        "Local Memory (Redis HNSW + MiniLM)",
  description: "Long-term memory + agent capability registry via local FastAPI sidecar + Redis.",
  kind:        "lifecycle",

  register(api) {
    const cfg    = api.pluginConfig ?? {};
    const log    = api.logger ?? console;
    const base   = (cfg.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    const limit  = cfg.memoryLimitNumber ?? DEFAULT_LIMIT;
    const tag    = "[memos-local]";

    // ── Helpers ─────────────────────────────────────────────────────────────
    const post = (path, body, timeout = RECALL_TIMEOUT) =>
      fetch(`${base}${path}`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
        signal:  AbortSignal.timeout(timeout),
      }).catch(e => {
        log.warn?.(`${tag} ${path} error: ${e?.message}`);
        return null;
      });

    /**
     * Normalize tool definitions from various OpenClaw/MCP formats
     * into the memory service ToolDefinition schema.
     */
    const normalizeTool = (t) => {
      if (!t) return null;
      const name = t.name || t.id || t.title || "";
      const description = t.description || t.desc || t.summary || "";
      if (!name) return null;

      // Detect source from tool metadata
      let source = "builtin";
      if (t.source === "mcp" || t.server || t.mcpServer) source = "mcp";
      else if (t.source === "plugin" || t.pluginId)      source = "plugin";
      else if (t.source === "skill"  || t.isSkill)       source = "skill";

      // Extract parameter names
      const parameters = [];
      if (t.inputSchema?.properties) {
        parameters.push(...Object.keys(t.inputSchema.properties));
      } else if (t.parameters) {
        parameters.push(...(Array.isArray(t.parameters)
          ? t.parameters.map(p => p.name || p)
          : Object.keys(t.parameters)));
      }

      return {
        name,
        description: description.slice(0, 400),
        category:    t.category || "",
        source,
        parameters:  parameters.slice(0, 10),
      };
    };

    /**
     * Extract environment state from the agent context.
     * Handles multiple possible context shapes from different OpenClaw versions.
     */
    const extractEnv = (ctx, event) => {
      const env = {};
      // Agent model info
      if (ctx?.modelId || ctx?.model)     env.agent_model   = ctx.modelId || ctx.model;
      if (ctx?.agentVersion)              env.agent_version = ctx.agentVersion;
      if (ctx?.sessionId || ctx?.sessionKey) {
        env.session_id = ctx.sessionId || ctx.sessionKey;
      }
      // Connected MCPs
      if (ctx?.mcpServers?.length)        env.active_mcps   = ctx.mcpServers.map(s => s.name || s.id || s);
      if (ctx?.plugins?.length)           env.active_plugins = ctx.plugins.map(p => p.id || p);
      if (ctx?.skills?.length)            env.active_skills  = ctx.skills.map(s => s.id || s.name || s);
      // Environment from event or system info
      if (event?.cwd || ctx?.cwd)         env.cwd           = event?.cwd || ctx?.cwd;
      if (event?.gitBranch || ctx?.gitBranch) env.git_branch = event?.gitBranch || ctx?.gitBranch;
      if (event?.gitRepo   || ctx?.gitRepo)   env.git_repo   = event?.gitRepo   || ctx?.gitRepo;
      // Runtime platform info (if available)
      if (typeof process !== "undefined") {
        env.runtime      = `node${process.version}`;
        env.os           = process.platform;
        env.os_version   = process.release?.name || "";
      }
      return env;
    };

    // ── before_agent_start: recall + register capability ────────────────────
    api.on("before_agent_start", async (event, ctx) => {
      const query = event?.prompt;

      // 1. Register tools (fire-and-forget — don't block agent start)
      const rawTools = event?.tools || ctx?.tools || ctx?.availableTools || [];
      const rawSkills = event?.skills || ctx?.skills || [];
      const allTools = [
        ...rawTools.map(t => normalizeTool(t)),
        ...rawSkills.map(s => normalizeTool({ ...s, source: "skill" })),
      ].filter(Boolean);

      if (allTools.length > 0) {
        post("/register-tools", {
          tools:    allTools,
          agent_id: ctx?.sessionId || ctx?.sessionKey || "",
        }, REG_TIMEOUT).then(res => {
          if (res?.ok) log.info?.(`${tag} registered ${allTools.length} tools`);
        });
      }

      // 2. Register environment state (fire-and-forget)
      const envState = extractEnv(ctx, event);
      if (Object.keys(envState).length > 0) {
        post("/register-env", envState, REG_TIMEOUT).catch(() => {});
      }

      // 3. Recall memories (blocking — result goes into prependContext)
      if (!query) return;

      try {
        const res = await fetch(`${base}/recall`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            query,
            session_id:          ctx?.sessionId ?? ctx?.sessionKey ?? "",
            memory_limit_number: limit,
            include_tools:       true,
            include_procedures:  true,
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

    // ── tool_end: ToolMem feedback — record success/fail per tool use ────────
    api.on("tool_end", async (event, ctx) => {
      const toolName = event?.toolName || event?.tool?.name || event?.name || "";
      if (!toolName) return;
      const sessionId = ctx?.sessionId ?? ctx?.sessionKey ?? "";

      // Heuristic success: failure if error/exception in output
      const output = String(event?.output || event?.result || event?.response || "");
      const errorWords = ["error", "exception", "traceback", "failed", "not found",
                          "permission denied", "timeout", "command not found"];
      const success = !errorWords.some(w => output.toLowerCase().includes(w));

      post("/tool-feedback", { tool_name: toolName, success, session_id: sessionId },
           REG_TIMEOUT).catch(() => {});
    });

    // ── agent_end: compact + store + promote Tier 1 → Tier 2 + TIG ─────────
    api.on("agent_end", async (event, ctx) => {
      if (!event?.success) return;
      const messages  = event?.messages;
      const sessionId = ctx?.sessionId ?? ctx?.sessionKey ?? "";
      if (!messages?.length) return;

      // 1. Queue turn storage (async background, non-blocking)
      fetch(`${base}/store`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ messages, session_id: sessionId }),
        signal:  AbortSignal.timeout(STORE_TIMEOUT),
      }).catch(() => {});

      if (sessionId) {
        // 2. Mid-session compact (v0.9.3): trim Tier 1 KV before promotion.
        fetch(`${base}/session/compact`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ session_id: sessionId, threshold_chars: 3000 }),
          signal:  AbortSignal.timeout(STORE_TIMEOUT),
        }).catch(() => {});

        // 3. Promote Tier 1 session KV → Tier 2 long-term memory.
        fetch(`${base}/session/compress`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ session_id: sessionId, wait: false }),
          signal:  AbortSignal.timeout(STORE_TIMEOUT),
        }).catch(() => {});

        // 4. AutoTool TIG (v0.9.5): extract tool call sequence from messages,
        //    record transitions into mem:tool_graph for next-tool suggestions.
        const toolSeq = [];
        for (const msg of messages) {
          const content = Array.isArray(msg?.content) ? msg.content
                        : (typeof msg?.content === "string" ? [] : []);
          for (const block of content) {
            if (block?.type === "tool_use" && block?.name) {
              toolSeq.push(block.name);
            }
          }
        }
        if (toolSeq.length >= 2) {
          fetch(`${base}/record-tool-sequence`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ sequence: toolSeq, session_id: sessionId }),
            signal:  AbortSignal.timeout(STORE_TIMEOUT),
          }).catch(() => {});
        }
      }
    });

    log.info?.(`${tag} v0.5.0 registered (ToolMem+TIG+MACLA, base=${base}, limit=${limit})`);
  },
};
