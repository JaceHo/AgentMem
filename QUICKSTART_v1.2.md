# AgentMem Quick Start - v1.2.0 (with Auto-Crystallization & User Feedback)

## What's New in v1.2.0?

🎉 **Automatic Crystallization**: Sessions are automatically distilled into lessons learned every 6 hours  
⭐ **User Feedback**: Rate memories to help the system learn your preferences  
📌 **Pin Important Facts**: Mark critical memories as permanent  
🗑️ **Delete Incorrect Facts**: Remove outdated or wrong memories instantly  

---

## Quick Start

### 1. Install & Setup

```bash
git clone https://github.com/JaceHo/AgentMem
cd AgentMem
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash agentmem.sh setup   # installs service + all hooks
bash agentmem.sh start   # starts Redis + AgentMem service
```

### 2. Verify Installation

```bash
curl http://localhost:18800/health
# Should return: {"status": "ok", "version": "1.2.0"}
```

### 3. Use with Claude Code

Open a new Claude Code session. Memory is now automatic!

Every prompt gets context like:
```xml
<cross_session_memory>
## Lessons Learned          ← NEW in v1.2.0!
- **Completed Session** (12 facts): Configured Redis cluster...
  Key entities: Redis, TLS, certificates

## Last Session Summary
Completed Redis migration for aiserv gateway...

## Long-Term Memory (Facts)
1. [rule] Always use type hints in Python functions
2. [preference] Jace prefers bun over npm
</cross_session_memory>
```

---

## User Feedback Features

### Rate a Memory

When you see a memory in context, you can rate it:

```bash
# Rate as helpful (5 stars)
curl -X POST http://localhost:18800/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "element_id": "01HX...",  # from fact metadata
    "rating": 5,
    "comment": "Very helpful!"
  }'

# Rate as unhelpful (1 star)
curl -X POST http://localhost:18800/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "element_id": "01HX...",
    "rating": 1,
    "comment": "This is outdated"
  }'
```

**What happens:**
- ⭐⭐⭐⭐⭐ Rating 4-5: Importance boosted × 1.2, fact reinforced
- ⭐ Rating 1-2: Importance reduced × 0.7, flagged for review
- ⭐⭐⭐ Rating 3: No change, just recorded

### Pin Important Facts

Never let critical facts be pruned:

```bash
curl -X POST http://localhost:18800/facts/01HX.../pin
```

Pinned facts have `importance = 1.0` and are excluded from pruning.

### Delete Incorrect Facts

Remove wrong memories immediately:

```bash
curl -X DELETE http://localhost:18800/facts/01HX...
```

⚠️ **Warning**: This is permanent! No undo.

### View Fact Metadata

See full details about a memory:

```bash
curl http://localhost:18800/facts/01HX.../metadata
```

Returns:
```json
{
  "element_id": "01HX...",
  "content": "Always use type hints in Python functions",
  "category": "rule",
  "importance": 0.85,
  "confidence": 0.9,
  "effective_confidence": 0.87,
  "user_rating": 5,
  "user_comment": "Very helpful guideline!",
  "pinned": false,
  "access_count": 12,
  "source_count": 3,
  ...
}
```

---

## Automatic Crystallization

### How It Works

Every 6 hours, AgentMem automatically:
1. Scans for completed sessions (age > 24h, >5 facts)
2. Distills them into structured digests:
   - What was accomplished?
   - What was learned?
   - What entities were involved?
3. Stores digests with 90-day TTL
4. Includes top-3 digests in recall context under "## Lessons Learned"

### Manual Crystallization

Trigger crystallization on demand:

```bash
curl -X POST http://localhost:18800/crystallize \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "ses_123",
    "max_facts": 20
  }'
```

### View Crystallized Digests

```bash
# Get digest for specific session
curl http://localhost:18800/crystallize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "ses_123"}'

# Or check Redis directly
redis-cli KEYS "mem:crystallized:*"
redis-cli GET "mem:crystallized:ses_123"
```

---

## Monitoring & Health Checks

### Check Service Status

```bash
bash agentmem.sh status
# or
curl http://localhost:18800/health
```

### View Memory Statistics

```bash
curl http://localhost:18800/stats
```

Returns:
```json
{
  "facts": 2450,
  "episodes": 1334,
  "procedures": 28664,
  "tools": 84,
  "crystallized_sessions": 45  ← NEW!
}
```

### View Lifecycle Stats

```bash
curl http://localhost:18800/lifecycle/stats
```

Returns confidence distribution, supersession counts, category health.

### Trigger Consolidation Manually

```bash
# Async (background)
curl -X POST http://localhost:18800/consolidate

# Sync (wait for results)
curl -X POST http://localhost:18800/consolidate/sync
```

### Trigger Hard Prune

```bash
curl -X POST http://localhost:18800/consolidate/hard-prune
```

Physically removes superseded facts (>7 days old) and stale episodes (>180 days).

---

## Web Dashboard

Open http://localhost:3113 to view:
- Real-time memory activity
- Search memories
- View knowledge graph
- Monitor consolidation
- **NEW**: View user feedback ratings

---

## Troubleshooting

### Memories Not Appearing in Context

1. Check if facts were stored:
   ```bash
   curl http://localhost:18800/stats
   ```

2. Test recall manually:
   ```bash
   curl -X POST http://localhost:18800/recall \
     -H "Content-Type: application/json" \
     -d '{
       "query": "your query here",
       "session_id": "test"
     }'
   ```

3. Check Redis connection:
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

### Crystallization Not Running

1. Check background tasks in logs:
   ```bash
   tail -f ~/.agentmem/logs/dashboard.jsonl | grep crystallize
   ```

2. Manually trigger:
   ```bash
   curl -X POST http://localhost:18800/crystallize \
     -H "Content-Type: application/json" \
     -d '{"session_id": "ses_123"}'
   ```

### User Feedback Not Working

1. Verify element_id is correct:
   ```bash
   curl http://localhost:18800/facts/01HX.../metadata
   ```

2. Check rating is 1-5:
   ```bash
   # Valid ratings: 1, 2, 3, 4, 5
   ```

---

## Migration from v1.1.x

No migration needed! All improvements are backward compatible.

New features activate automatically:
- ✅ Crystallization starts running within 6 hours of upgrade
- ✅ Feedback endpoints available immediately
- ✅ Existing facts work with new lifecycle management

---

## Performance Impact

New features have minimal performance impact:
- Crystallization: Runs every 6h in background (~2-5s per session)
- User feedback: O(1) Redis operations (<1ms)
- Recall latency: Unchanged (15ms P50)

---

## Next Steps

1. **Start using AgentMem** with your AI agents
2. **Rate memories** as you use them to train the system
3. **Pin critical facts** that should never be pruned
4. **Monitor crystallization** via dashboard or logs
5. **Check back in Phase 2** for anomaly detection and explainability features

---

## Support

- 📖 Full documentation: [`README.md`](file:///Users/jace/code/agentmem/README.md)
- 🔍 Architecture audit: [`AUDIT_AND_IMPROVEMENTS.md`](file:///Users/jace/code/agentmem/AUDIT_AND_IMPROVEMENTS.md)
- 📊 Improvements summary: [`IMPROVEMENTS_SUMMARY.md`](file:///Users/jace/code/agentmem/IMPROVEMENTS_SUMMARY.md)
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

**Enjoy your intelligent, self-improving memory system!** 🎉
